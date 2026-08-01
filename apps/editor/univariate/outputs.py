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
from dz_lib import univariate, bivariate, config as dz_config
from dz_lib.bivariate.distributions import *
from dz_lib.univariate import mds, unmix, distributions, mda, metrics, histograms
from dz_lib.univariate.mda import core as mda_core
from dz_lib.univariate.mda import mla as mda_mla
from dz_lib.utils import data, matrices
import numpy as np
import pandas as pd


def _configure_dz_lib(project):
    """Configure dz_lib with project's sigma settings."""
    sigma_in = project.settings.uncertainty_settings.sigma_in
    sigma_out = project.settings.uncertainty_settings.sigma_out
    dz_config.set_sigma_in(sigma_in)
    dz_config.set_sigma_out(sigma_out)


def _grains_to_mda_sample(grains, sigma_in):
    """Convert legacy Grain list to new MDA Sample object."""
    ages = np.array([g.age for g in grains])
    errs = np.abs(np.array([g.uncertainty for g in grains]))

    # Filter out grains with non-finite values
    valid_mask = np.isfinite(ages) & np.isfinite(errs)
    ages = ages[valid_mask]
    errs = errs[valid_mask]

    if len(ages) == 0:
        raise ValueError("No valid grains found")

    # Replace zero uncertainties with a small default (1% of age or 1 Ma, whichever is larger)
    zero_mask = errs <= 0
    if np.any(zero_mask):
        default_errs = np.maximum(ages[zero_mask] * 0.01, 1.0)
        errs[zero_mask] = default_errs

    return mda_core.Sample(ages, errs, sigma_in=sigma_in)


def _mda_results_to_table(results, sigma_out):
    """Create a comparison table from MDA results dict."""
    rows = []
    for metric_name, result in results.items():
        if np.isfinite(result.mda):
            # Convert 1-sigma uncertainty to requested sigma level
            unc_1s = result.unc_1s if np.isfinite(result.unc_1s) else float('nan')
            unc_out = unc_1s * sigma_out if np.isfinite(unc_1s) else float('nan')
            rows.append({
                'Metric': metric_name,
                'MDA (Ma)': round(result.mda, 2) if np.isfinite(result.mda) else '',
                f'{sigma_out}σ (Myr)': round(unc_out, 2) if np.isfinite(unc_out) else '',
                'n': result.n_used,
                'MSWD': round(result.mswd, 2) if np.isfinite(result.mswd) else '',
            })
        else:
            rows.append({
                'Metric': metric_name,
                'MDA (Ma)': '',
                f'{sigma_out}σ (Myr)': '',
                'n': '',
                'MSWD': '',
            })
    df = pd.DataFrame(rows)
    df = df.set_index('Metric')
    df.index.name = None  # Remove index name so "Metric" doesn't appear on separate row
    return df


def _mda_results_to_graph(results, sigma_out, title, font_path, font_size, fig_width, fig_height):
    """Create a comparison graph from MDA results dict."""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.lines import Line2D

    methods = list(results.keys())
    ages = []
    uncertainties = []

    for metric_name, result in results.items():
        if np.isfinite(result.mda):
            ages.append(result.mda)
            unc_1s = result.unc_1s if np.isfinite(result.unc_1s) else float('nan')
            uncertainties.append(unc_1s)
        else:
            ages.append(float('nan'))
            uncertainties.append(float('nan'))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    x = np.arange(len(methods))

    color_1s = "black"
    color_2s = "cornflowerblue"

    for i in range(len(methods)):
        if np.isfinite(uncertainties[i]):
            ax.vlines(x[i], ages[i] - uncertainties[i] * 2, ages[i] + uncertainties[i] * 2,
                      color=color_2s, linewidth=5)
            ax.vlines(x[i], ages[i] - uncertainties[i], ages[i] + uncertainties[i],
                      color=color_1s, linewidth=5)

    ax.scatter(x, ages, color='white', edgecolor='black', s=100, zorder=3, marker='s')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_xlabel('Method', fontsize=font_size)
    ax.set_ylabel('Age (Ma)', fontsize=font_size)

    if title:
        if font_path:
            font_prop = fm.FontProperties(fname=font_path)
        else:
            font_prop = None
        ax.set_title(title, fontsize=font_size * 1.5, fontproperties=font_prop)

    legend_elements = [
        Line2D([0], [0], color=color_2s, lw=5, label='2σ'),
        Line2D([0], [0], color=color_1s, lw=5, label='1σ')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)

    fig.tight_layout()
    plt.close()

    return fig


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
            _configure_dz_lib(project)  # Apply project sigma settings
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

                points, kruskal_stress, dissimilarity_matrix, mds_embedding, mds_result = mds.mds_function(
                    samples=adjusted_samples,
                    metric='similarity',
                    non_metric=non_metric
                )
                pending_outputs = []
                if "mds_plot" in output_types:
                    graph_fig = mds.mds_graph(
                        points=points,
                        title=f"{output_title} (metric='similarity', stress={round(kruskal_stress, 2)})",
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
                        embedding=mds_embedding,
                        mds_result=mds_result,
                        kruskal_stress=kruskal_stress,
                        non_metric=non_metric,
                        title=f"{output_title} (metric='similarity', stress={round(kruskal_stress, 2)})",
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
            _configure_dz_lib(project)  # Apply project sigma settings
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
            _configure_dz_lib(project)  # Apply project sigma settings
            if request.method == "GET":
                output_title = request.args.get("outputTitle", "")
                output_types = request.args.getlist("outputType")
                sample_names = request.args.getlist("sampleNames")

                # Parse new MDA parameters
                selected_metrics = request.args.getlist("metrics")
                preset = request.args.get("preset", "harmonized")
                rank_by = request.args.get("rank_by", "age+1s")
                yc_min_n = int(request.args.get("yc_min_n", 2))
                ysp_target_mswd = float(request.args.get("ysp_target_mswd", 1.0))
                ysp_entry_rule = request.args.get("ysp_entry_rule", "global")

                spreadsheet_data = spreadsheet.text_to_array(project.data)
                loaded_samples = data.read_1d_samples(spreadsheet_data)
                active_samples = []
                for sample in loaded_samples:
                    sample.name = __clean_sample_name(sample.name)
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                sample = active_samples[0]

                sigma_in = project.settings.uncertainty_settings.sigma_in
                sigma_out = project.settings.uncertainty_settings.sigma_out

                # Build overrides for custom options
                overrides = {}
                if rank_by:
                    overrides['ysg'] = {'rank_by': rank_by}
                    overrides['y3za'] = {'rank_by': rank_by}
                if yc_min_n:
                    overrides['yc1s'] = {'min_n': yc_min_n, 'rank_by': rank_by}
                    overrides['yc2s'] = {'min_n': max(yc_min_n, 3), 'rank_by': rank_by}
                if ysp_target_mswd or ysp_entry_rule:
                    overrides['ysp'] = {
                        'target_mswd': ysp_target_mswd,
                        'entry_rule': ysp_entry_rule,
                        'rank_by': rank_by
                    }

                # Convert grains to MDA Sample and run all metrics
                try:
                    mda_sample = _grains_to_mda_sample(sample.grains, sigma_in)
                    mda_results = mda_core.all_metrics(
                        mda_sample,
                        preset=preset,
                        overrides=overrides,
                        include=selected_metrics if selected_metrics else None
                    )
                except Exception as e:
                    return jsonify({"error": str(e)}), 400

                pending_outputs = []

                if "mda_table" in output_types:
                    matrix_df = _mda_results_to_table(mda_results, sigma_out)
                    output_id = secrets.token_hex(15)
                    output_data = embedding.embed_matrix(
                        dataframe=matrix_df, output_id=output_id, project_id=project_id,
                        download_formats=['xlsx', 'xls', 'csv']
                    )
                    pending_outputs.append({"output_id": output_id, "output_type": "matrix", "output_data": output_data})

                if "mda_graph" in output_types:
                    graph_fig = _mda_results_to_graph(
                        mda_results, sigma_out, output_title,
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

                if "radial_plot" in output_types:
                    # Get MLA result for radial plot
                    mla_result = mda_results.get('MLA')
                    mla_grain = None
                    if mla_result and np.isfinite(mla_result.mda):
                        from dz_lib.univariate.data import Grain
                        mla_grain = Grain(mla_result.mda, mla_result.unc_1s * sigma_out)

                    graph_fig = mda_mla.radial_plot(
                        grains=sample.grains,
                        mla_result=mla_grain,
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
