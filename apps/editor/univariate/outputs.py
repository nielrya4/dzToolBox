"""
Univariate outputs routes - all output generation and management
"""

import secrets
from flask import request, jsonify, session
from flask_login import login_required
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2_fragments import render_block
from server import database
from utils import spreadsheet, compression
from utils.output import Output
from utils.project import project_from_json
from utils import embedding
from utils import monte_carlo_optimized
from dz_lib import univariate, bivariate
from dz_lib.bivariate.distributions import *
from dz_lib.univariate import mds, unmix, distributions, mda, metrics, histograms
from dz_lib.utils import data, matrices
import numpy as np

try:
    from celery_app import celery_app
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    celery_app = None

environment = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(("html", "jinja2"))
)


def __get_project(project_id):
    if session.get("open_project", 0) == project_id:
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        return project_from_json(project_content)
    else:
        return None


def __clean_sample_name(sample_name):
    try:
        num = float(sample_name)
        if num.is_integer():
            return str(int(num))
    except ValueError:
        pass
    return str(sample_name)


def register(app):

    @app.route('/projects/<int:project_id>/outputs', methods=['GET'])
    @login_required
    def get_project_outputs(project_id):
        if session.get("open_project", 0) == project_id:
            try:
                project = __get_project(project_id)
                return jsonify([{
                    'output_id': output.output_id,
                    'output_type': output.output_type,
                    'output_data': output.output_data
                } for output in project.outputs])
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/outputs/save', methods=['POST'])
    @login_required
    def save_output(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            outputs_data = request.get_json().get('outputs', [])
            for output_item in outputs_data:
                project.outputs.append(Output(
                    output_id=output_item['output_id'],
                    output_type=output_item['output_type'],
                    output_data=output_item['output_data']
                ))
            updated_project_content = project.to_json()
            compressed_proj_content = compression.compress(updated_project_content)
            database.write_file(project_id, compressed_proj_content)
            return render_block(
                environment=environment,
                template_name="editor/pages/univariate.html",
                block_name="outputs",
                outputs_data=project.outputs,
                project_id=project_id
            )
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/outputs/delete/<string:output_id>', methods=['POST'])
    @login_required
    def delete_output(project_id, output_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            project.delete_output(output_id)
            updated_project_content = project.to_json()
            compressed_proj_content = compression.compress(updated_project_content)
            database.write_file(project_id, compressed_proj_content)
            return render_block(
                environment=environment,
                template_name="editor/pages/univariate.html",
                block_name="outputs",
                outputs_data=project.outputs,
                project_id=project_id
            )
        else:
            return jsonify({"outputs": "access_denied"})

    @app.route('/projects/<int:project_id>/outputs/clear', methods=['POST'])
    @login_required
    def clear_outputs(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            project.outputs = []
            updated_project_content = project.to_json()
            compressed_proj_content = compression.compress(updated_project_content)
            database.write_file(project_id, compressed_proj_content)
            return render_block(
                environment=environment,
                template_name="editor/pages/univariate.html",
                block_name="outputs",
                outputs_data=project.outputs,
                project_id=project_id
            )
        else:
            return jsonify({"outputs": "access_denied"})

    @app.route('/projects/<int:project_id>/outputs/active-jobs', methods=['GET'])
    @login_required
    def get_active_jobs(project_id):
        if session.get("open_project", 0) == project_id:
            return jsonify({"active_jobs": []})
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/outputs/new/distribution', methods=['GET'])
    @login_required
    def new_distro(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            if request.method == "GET":
                output_title = request.args.get("outputTitle", None)
                output_type = request.args.get("outputType", "kde")
                sample_names = request.args.getlist("sampleNames")
                spreadsheet_data = spreadsheet.text_to_array(project.data)
                loaded_samples = data.read_1d_samples(spreadsheet_data)
                active_samples = []
                for sample in loaded_samples:
                    sample.name = __clean_sample_name(sample.name)
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                adjusted_samples = []
                for sample in active_samples:
                    if project.settings.statistical_settings.matrix_function_type == "kde" and output_type != "pdp":
                        sample.replace_grain_uncertainties(project.settings.statistical_settings.kde_bandwidth)
                    adjusted_samples.append(sample)
                adjusted_samples.reverse()

                if output_type == 'kde':
                    distros = [
                        univariate.distributions.kde_function(
                            sample=sample,
                            bandwidth=float(project.settings.statistical_settings.kde_bandwidth)
                        ).subset(project.settings.age_settings.min_age, project.settings.age_settings.max_age)
                        for sample in adjusted_samples
                    ]
                elif output_type == 'pdp':
                    distros = [univariate.distributions.pdp_function(sample) for sample in adjusted_samples]
                elif output_type == 'cdf':
                    distros = [
                        univariate.distributions.cdf_function(
                            univariate.distributions.kde_function(
                                sample=sample,
                                bandwidth=float(project.settings.statistical_settings.kde_bandwidth)
                            )
                        )
                        for sample in adjusted_samples
                    ]
                else:
                    raise ValueError("output_type is not supported")

                graph_fig = univariate.distributions.distribution_graph(
                    distributions=distros,
                    stacked=project.settings.graph_settings.stack_graphs == "true",
                    legend=project.settings.graph_settings.legend == "true",
                    title=output_title,
                    font_path=f'static/global/fonts/{project.settings.graph_settings.font_name}.ttf',
                    font_size=project.settings.graph_settings.font_size,
                    fig_width=project.settings.graph_settings.figure_width,
                    fig_height=project.settings.graph_settings.figure_height,
                    color_map=project.settings.graph_settings.color_map,
                    x_min=project.settings.age_settings.min_age,
                    x_max=project.settings.age_settings.max_age,
                    modes_labeled=project.settings.graph_settings.modes_labeled,
                    fill=project.settings.graph_settings.fill
                )
                output_id = secrets.token_hex(15)
                output_data = embedding.embed_graph(
                    fig=graph_fig,
                    output_id=output_id,
                    project_id=project_id,
                    fig_type="matplotlib",
                    img_format='svg',
                    download_formats=['svg', 'png']
                )
                return jsonify({"outputs": [{
                    "output_id": output_id,
                    "output_type": "graph",
                    "output_data": output_data
                }]})
            else:
                return jsonify({"outputs": "method not allowed"})
        else:
            return jsonify({"outputs": "access_denied"})

    @app.route('/projects/<int:project_id>/outputs/new/histogram', methods=['GET'])
    @login_required
    def new_histogram(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            if request.method == "GET":
                output_title = request.args.get("outputTitle", None)
                output_type = request.args.get("outputType", "histogram")
                sample_names = request.args.getlist("sampleNames")
                bin_edges_str = request.args.get("binEdges", "")
                bin_edges = [float(x.strip()) for x in bin_edges_str.split(",") if x.strip()]
                bin_labels_str = request.args.get("binLabels", "")
                bin_labels = [x.strip() for x in bin_labels_str.split(",") if x.strip()] if bin_labels_str else None

                spreadsheet_data = spreadsheet.text_to_array(project.data)
                loaded_samples = data.read_1d_samples(spreadsheet_data)
                active_samples = []
                for sample in loaded_samples:
                    sample.name = __clean_sample_name(sample.name)
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                active_samples.reverse()

                bin_list = histograms.BinList(
                    edges=bin_edges,
                    labels=bin_labels,
                    color_map=project.settings.graph_settings.color_map
                )

                if output_type == 'histogram':
                    graph_fig = histograms.histogram_graph(
                        samples=active_samples,
                        bin_list=bin_list,
                        legend=project.settings.graph_settings.legend == "true",
                        title=output_title,
                        font_path=f'static/global/fonts/{project.settings.graph_settings.font_name}.ttf',
                        font_size=project.settings.graph_settings.font_size,
                        fig_width=project.settings.graph_settings.figure_width,
                        fig_height=project.settings.graph_settings.figure_height,
                        color_map=project.settings.graph_settings.color_map,
                        fill=project.settings.graph_settings.fill == "true"
                    )
                elif output_type == 'pie_chart':
                    n_cols = int(request.args.get("nCols", 2))
                    graph_fig = histograms.histogram_pie_chart(
                        samples=active_samples,
                        bin_list=bin_list,
                        title=output_title,
                        font_path=f'static/global/fonts/{project.settings.graph_settings.font_name}.ttf',
                        font_size=project.settings.graph_settings.font_size,
                        fig_width=project.settings.graph_settings.figure_width,
                        n_cols=n_cols,
                    )
                else:
                    raise ValueError("output_type is not supported")

                output_id = secrets.token_hex(15)
                output_data = embedding.embed_graph(
                    fig=graph_fig,
                    output_id=output_id,
                    project_id=project_id,
                    fig_type="matplotlib",
                    img_format='svg',
                    download_formats=['svg', 'png']
                )
                return jsonify({"outputs": [{
                    "output_id": output_id,
                    "output_type": "graph",
                    "output_data": output_data
                }]})
            else:
                return jsonify({"outputs": "method not allowed"})
        else:
            return jsonify({"outputs": "access_denied"})

    @app.route('/projects/<int:project_id>/outputs/new/mds', methods=['GET'])
    @login_required
    def new_mds(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            if request.method == "GET":
                output_title = request.args.get("outputTitle", None)
                metric = request.args.get("metric", "similarity")
                non_metric = request.args.get("mds_type") == "non_metric"
                output_types = request.args.getlist("outputType")
                sample_names = request.args.getlist("sampleNames")
                spreadsheet_data = spreadsheet.text_to_array(project.data)
                loaded_samples = data.read_1d_samples(spreadsheet_data)
                active_samples = []
                for sample in loaded_samples:
                    sample.name = __clean_sample_name(sample.name)
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                adjusted_samples = []
                for sample in active_samples:
                    if project.settings.statistical_settings.matrix_function_type == "kde" and metric != "pdp_graph":
                        sample.replace_grain_uncertainties(project.settings.statistical_settings.kde_bandwidth)
                    adjusted_samples.append(sample)
                adjusted_samples.reverse()

                points, stress, dissimilarity_matrix, scaled_mds_result, mds_result = mds.mds_function(
                    samples=adjusted_samples,
                    metric='similarity',
                    non_metric=non_metric
                )
                pending_outputs = []
                if "mds_plot" in output_types:
                    graph_fig = mds.mds_graph(
                        points=points,
                        title=f"{output_title} (metric='similarity', stress={round(stress, 2)})",
                        font_path=f'static/global/fonts/{project.settings.graph_settings.font_name}.ttf',
                        font_size=project.settings.graph_settings.font_size,
                        fig_width=project.settings.graph_settings.figure_width,
                        fig_height=project.settings.graph_settings.figure_height,
                        color_map=project.settings.graph_settings.color_map
                    )
                    output_id = secrets.token_hex(15)
                    output_data = embedding.embed_graph(
                        fig=graph_fig, output_id=output_id, project_id=project_id,
                        fig_type="matplotlib", img_format='svg', download_formats=['svg', 'png']
                    )
                    pending_outputs.append({"output_id": output_id, "output_type": "graph", "output_data": output_data})
                if "shepard_plot" in output_types:
                    graph_fig = mds.shepard_plot(
                        dissimilarity_matrix=dissimilarity_matrix,
                        scaled_mds_result=scaled_mds_result,
                        mds_result=mds_result,
                        non_metric=non_metric,
                        title=f"{output_title} (metric='similarity', stress={round(stress, 2)})",
                        font_path=f'static/global/fonts/{project.settings.graph_settings.font_name}.ttf',
                        font_size=project.settings.graph_settings.font_size,
                        fig_width=project.settings.graph_settings.figure_width,
                        fig_height=project.settings.graph_settings.figure_height
                    )
                    output_id = secrets.token_hex(15)
                    output_data = embedding.embed_graph(
                        fig=graph_fig, output_id=output_id, project_id=project_id,
                        fig_type="matplotlib", img_format='svg', download_formats=['svg', 'png']
                    )
                    pending_outputs.append({"output_id": output_id, "output_type": "graph", "output_data": output_data})
                return jsonify({"outputs": pending_outputs})
            else:
                return jsonify({"outputs": "method not allowed"})
        else:
            return jsonify({"outputs": "access_denied"})

    @app.route('/projects/<int:project_id>/outputs/new/unmix', methods=['GET'])
    @login_required
    def new_unmix(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            if request.method == "GET":
                output_title = request.args.get("outputTitle", "")
                metric = request.args.get("metric", "cross_correlation")
                output_types = request.args.getlist("outputType")
                sample_names = request.args.getlist("sampleNames")
                spreadsheet_data = spreadsheet.text_to_array(project.data)
                loaded_samples = data.read_1d_samples(spreadsheet_data)
                active_samples = []
                for sample in loaded_samples:
                    sample.name = __clean_sample_name(sample.name)
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                adjusted_samples = []
                for sample in active_samples:
                    if project.settings.statistical_settings.matrix_function_type == "kde":
                        sample.replace_grain_uncertainties(project.settings.statistical_settings.kde_bandwidth)
                    adjusted_samples.append(sample)
                x_min = project.settings.age_settings.min_age
                x_max = project.settings.age_settings.max_age
                sample_pdps = [univariate.distributions.pdp_function(sample, x_min, x_max) for sample in adjusted_samples]
                if metric == 'cross_correlation':
                    sink_distribution = sample_pdps[0]
                    source_distributions = sample_pdps[1:]
                else:
                    sink_distribution = univariate.distributions.cdf_function(sample_pdps[0])
                    source_distributions = [univariate.distributions.cdf_function(p) for p in sample_pdps[1:]]
                contributions, stdevs, top_distributions = (
                    monte_carlo_optimized.monte_carlo_model_optimized(
                        sink_distribution=sink_distribution,
                        source_distributions=source_distributions,
                        n_trials=int(project.settings.statistical_settings.n_unmix_trials),
                        metric=metric
                    )
                )
                contribution_pairs = [
                    unmix.Contribution(name=active_samples[1:][i].name, contribution=contributions[i], standard_deviation=stdevs[i])
                    for i in range(len(active_samples[1:]))
                ]
                pending_outputs = []
                if "contribution_table" in output_types:
                    matrix_df = univariate.unmix.relative_contribution_table(contributions=contribution_pairs, metric=metric)
                    output_id = secrets.token_hex(15)
                    output_data = embedding.embed_matrix(
                        dataframe=matrix_df, output_id=output_id, project_id=project_id,
                        download_formats=['xlsx', 'xls', 'csv']
                    )
                    pending_outputs.append({"output_id": output_id, "output_type": "matrix", "output_data": output_data})
                if "contribution_graph" in output_types:
                    graph_fig = univariate.unmix.relative_contribution_graph(
                        contributions=contribution_pairs,
                        title=f"{output_title} (metric='{metric}')",
                        font_path=f'static/global/fonts/{project.settings.graph_settings.font_name}.ttf',
                        font_size=project.settings.graph_settings.font_size,
                        fig_width=project.settings.graph_settings.figure_width,
                        fig_height=project.settings.graph_settings.figure_height
                    )
                    output_id = secrets.token_hex(15)
                    output_data = embedding.embed_graph(
                        fig=graph_fig, output_id=output_id, project_id=project_id,
                        fig_type="matplotlib", img_format='svg', download_formats=['svg', 'png']
                    )
                    pending_outputs.append({"output_id": output_id, "output_type": "graph", "output_data": output_data})
                if "trials_graph" in output_types:
                    r2_vals = [metrics.r2(top_distro.y_values, sink_distribution.y_values) for top_distro in top_distributions]
                    avg_r2 = np.average(r2_vals)
                    output_title += f" (r^2={round(avg_r2, 3)}) (metric='{metric}')"
                    graph_fig = univariate.unmix.top_trials_graph(
                        sink_distribution=sink_distribution,
                        model_distributions=top_distributions,
                        x_min=x_min, x_max=x_max,
                        title=output_title,
                        font_path=f'static/global/fonts/{project.settings.graph_settings.font_name}.ttf',
                        font_size=project.settings.graph_settings.font_size,
                        fig_width=project.settings.graph_settings.figure_width,
                        fig_height=project.settings.graph_settings.figure_height
                    )
                    output_id = secrets.token_hex(15)
                    output_data = embedding.embed_graph(
                        fig=graph_fig, output_id=output_id, project_id=project_id,
                        fig_type="matplotlib", img_format='svg', download_formats=['svg', 'png']
                    )
                    pending_outputs.append({"output_id": output_id, "output_type": "graph", "output_data": output_data})
                return jsonify({"outputs": pending_outputs})
            else:
                return jsonify({"outputs": "method not allowed"})
        else:
            return jsonify({"outputs": "access_denied"})

    @app.route('/projects/<int:project_id>/outputs/new/matrix', methods=['GET'])
    @login_required
    def new_matrix(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            if request.method == "GET":
                output_title = request.args.get("outputTitle", None)
                output_type = request.args.get("outputType", "kde")
                sample_names = request.args.getlist("sampleNames")
                spreadsheet_data = spreadsheet.text_to_array(project.data)
                loaded_samples = data.read_1d_samples(spreadsheet_data)
                active_samples = []
                for sample in loaded_samples:
                    sample.name = __clean_sample_name(sample.name)
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                adjusted_samples = []
                for sample in active_samples:
                    if project.settings.statistical_settings.matrix_function_type == "kde":
                        sample.replace_grain_uncertainties(project.settings.statistical_settings.kde_bandwidth)
                    adjusted_samples.append(sample)
                adjusted_samples.reverse()
                matrix_df = matrices.generate_data_frame(samples=adjusted_samples, metric=output_type)
                output_id = secrets.token_hex(15)
                output_data = embedding.embed_matrix(
                    dataframe=matrix_df, output_id=output_id, title=output_title,
                    project_id=project_id, download_formats=['xlsx', 'xls', 'csv']
                )
                return jsonify({"outputs": [{
                    "output_id": output_id, "output_type": "matrix", "output_data": output_data
                }]})
            else:
                return jsonify({"outputs": "method not allowed"})
        else:
            return jsonify({"outputs": "access_denied"})

    @app.route('/projects/<int:project_id>/outputs/new/2d-distribution', methods=['GET'])
    @login_required
    def new_2d_distro(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            if request.method == "GET":
                output_title = request.args.get("outputTitle", None)
                output_type = request.args.get("outputType", "kde")
                sample_names = request.args.getlist("sampleNames")
                spreadsheet_data = spreadsheet.text_to_array(project.data)
                loaded_samples = data.read_2d_samples(spreadsheet_data)
                active_samples = []
                for sample in loaded_samples:
                    sample.name = __clean_sample_name(sample.name)
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                bivariate_distro = kde_function_2d(active_samples[0])
                if output_type == 'kde_2d_surface':
                    fig_type = "plotly"
                    graph_fig = kde_graph_2d(
                        bivariate_distro=bivariate_distro, title=output_title,
                        font_path=f'static/global/fonts/{project.settings.graph_settings.font_name}.ttf',
                        font_size=project.settings.graph_settings.font_size,
                        fig_width=project.settings.graph_settings.figure_width,
                        fig_height=project.settings.graph_settings.figure_height
                    )
                    img_format = 'png'
                elif output_type == 'kde_2d_heatmap':
                    fig_type = "matplotlib"
                    graph_fig = heatmap(
                        bivariate_distro=bivariate_distro, show_points=True, title=output_title,
                        font_path=f'static/global/fonts/{project.settings.graph_settings.font_name}.ttf',
                        font_size=project.settings.graph_settings.font_size,
                        color_map=project.settings.graph_settings.color_map,
                        fig_width=project.settings.graph_settings.figure_width,
                        fig_height=project.settings.graph_settings.figure_height
                    )
                    img_format = 'png'
                else:
                    raise ValueError("output_type is not supported")
                output_id = secrets.token_hex(15)
                output_data = embedding.embed_graph(
                    fig=graph_fig, output_id=output_id, project_id=project_id,
                    fig_type=fig_type, img_format=img_format, download_formats=['svg', 'png']
                )
                return jsonify({"outputs": [{
                    "output_id": output_id, "output_type": "graph", "output_data": output_data
                }]})
            else:
                return jsonify({"outputs": "method not allowed"})
        else:
            return jsonify({"outputs": "access_denied"})

    @app.route('/projects/<int:project_id>/outputs/new/mda', methods=['GET'])
    @login_required
    def new_mda(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            if request.method == "GET":
                output_title = request.args.get("outputTitle", "")
                output_types = request.args.getlist("outputType")
                sample_names = request.args.getlist("sampleNames")
                spreadsheet_data = spreadsheet.text_to_array(project.data)
                loaded_samples = data.read_1d_samples(spreadsheet_data)
                active_samples = []
                for sample in loaded_samples:
                    sample.name = __clean_sample_name(sample.name)
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                sample = active_samples[0]
                pending_outputs = []
                if "mda_table" in output_types:
                    matrix_df = univariate.mda.comparison_table(sample.grains)
                    output_id = secrets.token_hex(15)
                    output_data = embedding.embed_matrix(
                        dataframe=matrix_df, output_id=output_id, project_id=project_id,
                        download_formats=['xlsx', 'xls', 'csv']
                    )
                    pending_outputs.append({"output_id": output_id, "output_type": "matrix", "output_data": output_data})
                if "mda_graph" in output_types:
                    graph_fig = univariate.mda.comparison_graph(
                        grains=sample.grains, title=output_title,
                        font_path=f'static/global/fonts/{project.settings.graph_settings.font_name}.ttf',
                        font_size=project.settings.graph_settings.font_size,
                        fig_width=project.settings.graph_settings.figure_width,
                        fig_height=project.settings.graph_settings.figure_height
                    )
                    output_id = secrets.token_hex(15)
                    output_data = embedding.embed_graph(
                        fig=graph_fig, output_id=output_id, project_id=project_id,
                        fig_type="matplotlib", img_format='svg', download_formats=['svg', 'png']
                    )
                    pending_outputs.append({"output_id": output_id, "output_type": "graph", "output_data": output_data})
                if "rank_plot" in output_types:
                    graph_fig = univariate.mda.ranked_ages_plot(
                        grains=sample.grains, title=output_title,
                        x_min=project.settings.age_settings.min_age,
                        x_max=project.settings.age_settings.max_age,
                        font_path=f'static/global/fonts/{project.settings.graph_settings.font_name}.ttf',
                        font_size=project.settings.graph_settings.font_size,
                        fig_width=project.settings.graph_settings.figure_width,
                        fig_height=project.settings.graph_settings.figure_height
                    )
                    output_id = secrets.token_hex(15)
                    output_data = embedding.embed_graph(
                        fig=graph_fig, output_id=output_id, project_id=project_id,
                        fig_type="matplotlib", img_format='svg', download_formats=['svg', 'png']
                    )
                    pending_outputs.append({"output_id": output_id, "output_type": "graph", "output_data": output_data})
                if "ygf_graph" in output_types:
                    distro = distributions.pdp_function(sample)
                    fitted_grain, fitted_distro = mda.youngest_gaussian_fit(sample.grains)
                    graph_fig = distributions.distribution_graph(
                        distributions=[distro, fitted_distro], color_map="rainbow",
                        x_min=project.settings.age_settings.min_age,
                        x_max=project.settings.age_settings.max_age,
                        title=output_title,
                        font_path=f'static/global/fonts/{project.settings.graph_settings.font_name}.ttf',
                        font_size=project.settings.graph_settings.font_size,
                        fig_width=project.settings.graph_settings.figure_width,
                        fig_height=project.settings.graph_settings.figure_height
                    )
                    output_id = secrets.token_hex(15)
                    output_data = embedding.embed_graph(
                        fig=graph_fig, output_id=output_id, project_id=project_id,
                        fig_type="matplotlib", img_format='svg', download_formats=['svg', 'png']
                    )
                    pending_outputs.append({"output_id": output_id, "output_type": "graph", "output_data": output_data})
                return jsonify({"outputs": pending_outputs})
            else:
                return jsonify({"outputs": "method not allowed"})
        else:
            return jsonify({"outputs": "access_denied"})
