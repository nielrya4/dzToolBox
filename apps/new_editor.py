from flask import render_template, request, jsonify, session
from flask_login import login_required, current_user
from server import database
from utils import spreadsheet, compression
from utils.output import Output
from utils.project import project_from_json
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2_fragments import render_block
import json
import zlib
import secrets
import base64
from dz_lib import univariate, bivariate
from dz_lib.bivariate.distributions import *
from dz_lib.univariate import mds, unmix
from dz_lib.utils import data, matrices
from utils import embedding
environment = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(("html", "jinja2"))
)

def register(app):
    @app.route('/projects/<int:project_id>', methods=['GET'])
    @login_required
    def open_project(project_id):
        session["open_project"] = project_id
        file = database.get_file(project_id)
        project = __get_project(project_id)
        project_author = f"<User {file.user_id}>"
        if str(project_author) == str(current_user):
            spreadsheet_data = json.loads(project.data)
            loaded_samples = spreadsheet.read_samples(spreadsheet_data)
            samples_data = []
            for sample in loaded_samples:
                active = request.form.get(sample.name) == "true"
                samples_data.append([sample.name, active])
            if not project.outputs:
                project.outputs = [Output("Default", "graph", "<h1>No Outputs Yet</h1>")]
            return render_template("editor/editor.html",
                                   spreadsheet_data=project.data,
                                   samples=samples_data,
                                   outputs_data=project.outputs,
                                   project_id=project_id)
        else:
            return render_template("errors/403.html")

    @app.route('/projects/<int:project_id>/save', methods=['POST'])
    @login_required
    def save_project(project_id):
        if session.get("open_project", 0) == project_id:
            try:
                compressed_data = request.get_json().get('compressedData', '')
                if not compressed_data:
                    return jsonify({"success": False, "error": "No compressed data provided"})
                compressed_data_bytes = base64.b64decode(compressed_data)
                decompressed_data = zlib.decompress(compressed_data_bytes).decode('utf-8')
                json_data = json.loads(decompressed_data)
                data = json_data.get("data", [])
                if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
                    raise ValueError("Data is not in the expected list of lists format.")
                for i, row in enumerate(data):
                    for j, cell in enumerate(row):
                        if cell is not None:
                            if str(cell).strip() == '':
                                data[i][j] = None
                            elif is_float(cell):
                                data[i][j] = float(str(cell).strip())
                project = __get_project(project_id)
                project.data = spreadsheet.array_to_text(data)
                compressed_proj_content = compression.compress(project.to_json())
                database.write_file(project_id, compressed_proj_content)
                return jsonify({"success": True})
            except Exception as e:
                print(f"Error updating project data: {e}")
                return jsonify({"success": False, "error": str(e)})
        else:
            return jsonify({"save": "access_denied"})

    @app.route('/projects/<int:project_id>/sample-names', methods=['GET'])
    @login_required
    def get_sample_names(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            spreadsheet_data = spreadsheet.text_to_array(project.data)
            try:
                samples = spreadsheet.read_samples(spreadsheet_data)
                sample_names = [sample.name for sample in samples]
                return jsonify({"sample_names": sample_names})
            except Exception as e:
                print(f"Error updating project data: {e}")
                return jsonify({"success": False, "error": str(e)})
        else:
            return jsonify({"sample-names": "access_denied"})

    @app.route('/projects/<int:project_id>/settings', methods=['GET', 'POST'])
    @login_required
    def settings(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            if request.method == "POST":
                proj_settings = request.get_json()
                project.settings.from_json(proj_settings)
                updated_project_content = project.to_json()
                compressed_proj_content = compression.compress(updated_project_content)
                database.write_file(project_id, compressed_proj_content)
                return jsonify({"result": "ok"})
            elif request.method == "GET":
                settings = project.settings.to_json()
                return jsonify({"settings": settings})
            else:
                return jsonify({"settings": "method_not_allowed"})
        else:
            return jsonify({"settings": "access_denied"})


    @app.route('/projects/<int:project_id>/outputs/delete/<string:output_id>', methods=['POST'])
    @login_required
    def delete_output(project_id, output_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            if request.method == "POST":
                project.delete_output(output_id)
                updated_project_content = project.to_json()
                compressed_proj_content = compression.compress(updated_project_content)
                database.write_file(project_id, compressed_proj_content)
            return render_block(
                environment=environment,
                template_name="editor/editor.html",
                block_name="outputs",
                outputs_data=project.outputs,
                project_id=project_id
            )
        else:
            return jsonify({"outputs": "access_denied"})


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
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                x_min = data.get_x_min(active_samples)
                x_max = data.get_x_max(active_samples)
                adjusted_samples = []
                for sample in active_samples:
                    if project.settings.matrix_function_type == "kde" and output_type != "pdp_graph":
                        sample.replace_grain_uncertainties(project.settings.kde_bandwidth)
                    adjusted_samples.append(sample)
                adjusted_samples.reverse()
                if output_type == 'kde':
                    distributions = []
                    for sample in adjusted_samples:
                        distributions.append(
                            univariate.distributions.kde_function(
                                sample=sample,
                                bandwidth=float(project.settings.kde_bandwidth),
                                x_min=x_min,
                                x_max=x_max
                            )
                        )
                    graph_fig = univariate.distributions.distribution_graph(
                        distributions=distributions,
                        stacked=project.settings.stack_graphs == "true",
                        legend=project.settings.legend == "true",
                        title=output_title,
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
                        font_size=project.settings.font_size,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height,
                        color_map=project.settings.color_map
                    )
                elif output_type == 'pdp':
                    distributions = []
                    for sample in adjusted_samples:
                        distributions.append(
                            univariate.distributions.pdp_function(
                                sample=sample,
                                x_min=x_min,
                                x_max=x_max
                            )
                        )
                    graph_fig = univariate.distributions.distribution_graph(
                        distributions=distributions,
                        stacked=project.settings.stack_graphs == "true",
                        legend=project.settings.legend == "true",
                        title=output_title,
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
                        font_size=project.settings.font_size,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height,
                        color_map=project.settings.color_map
                    )
                elif output_type == 'cdf':
                    distributions = []
                    for sample in adjusted_samples:
                        distributions.append(
                            univariate.distributions.cdf_function(
                                univariate.distributions.kde_function(
                                    sample=sample,
                                    bandwidth=float(project.settings.kde_bandwidth),
                                    x_min=x_min,
                                    x_max=x_max
                                )
                            )
                        )
                    graph_fig = univariate.distributions.distribution_graph(
                        distributions=distributions,
                        stacked=project.settings.stack_graphs == "true",
                        legend=project.settings.legend == "true",
                        title=output_title,
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
                        font_size=project.settings.font_size,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height,
                        color_map=project.settings.color_map
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
                    download_formats=['svg', 'png', 'jpg', 'pdf', 'eps', 'webp']
                )
                new_output = Output(
                    output_id=output_id,
                    output_type="graph",
                    output_data=output_data
                )
                project.outputs.append(new_output)
                updated_project_content = project.to_json()
                compressed_proj_content = compression.compress(updated_project_content)
                database.write_file(project_id, compressed_proj_content)
                return render_block(
                    environment=environment,
                    template_name="editor/editor.html",
                    block_name="outputs",
                    outputs_data=project.outputs,
                    project_id=project_id
                )
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
                output_type = request.args.get("outputType", "kde")
                sample_names = request.args.getlist("sampleNames")
                spreadsheet_data = spreadsheet.text_to_array(project.data)
                loaded_samples = data.read_1d_samples(spreadsheet_data)
                active_samples = []
                for sample in loaded_samples:
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                adjusted_samples = []
                for sample in active_samples:
                    if project.settings.matrix_function_type == "kde" and output_type != "pdp_graph":
                        sample.replace_grain_uncertainties(project.settings.kde_bandwidth)
                    adjusted_samples.append(sample)
                adjusted_samples.reverse()
                if output_type == 'mds_similarity':
                    points, stress = mds.mds_function(
                        samples=adjusted_samples,
                        metric='similarity'
                    )
                    graph_fig = mds.mds_graph(
                        points=points,
                        title=f"{output_title} (metric='similarity', stress={stress})",
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
                        font_size=project.settings.font_size,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height,
                        color_map=project.settings.color_map
                    )
                elif output_type == 'mds_likeness':
                    points, stress = mds.mds_function(
                        samples=adjusted_samples,
                        metric='likeness'
                    )
                    graph_fig = mds.mds_graph(
                        points=points,
                        title=f"{output_title} (metric='likeness', stress={stress})",
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
                        font_size=project.settings.font_size,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height,
                        color_map=project.settings.color_map
                    )
                elif output_type == 'mds_cross_correlation':
                    points, stress = mds.mds_function(
                        samples=adjusted_samples,
                        metric='cross_correlation'
                    )
                    graph_fig = mds.mds_graph(
                        points=points,
                        title=f"{output_title} (metric='cross_correlation', stress={stress}))",
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
                        font_size=project.settings.font_size,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height,
                        color_map=project.settings.color_map
                    )
                elif output_type == 'mds_ks':
                    points, stress = mds.mds_function(
                        samples=adjusted_samples,
                        metric='ks'
                    )
                    graph_fig = mds.mds_graph(
                        points=points,
                        title=f"{output_title} (metric='ks', stress={stress})",
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
                        font_size=project.settings.font_size,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height,
                        color_map=project.settings.color_map
                    )
                elif output_type == 'mds_kuiper':
                    points, stress = mds.mds_function(
                        samples=adjusted_samples,
                        metric='kuiper'
                    )
                    graph_fig = mds.mds_graph(
                        points=points,
                        title=f"{output_title} (metric='kuiper', stress={stress})",
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
                        font_size=project.settings.font_size,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height,
                        color_map=project.settings.color_map
                    )
                else:
                    raise ValueError(f"output_type '{output_type}' is not supported")
                output_id = secrets.token_hex(15)
                output_data = embedding.embed_graph(
                    fig=graph_fig,
                    output_id=output_id,
                    project_id=project_id,
                    fig_type="matplotlib",
                    img_format='svg',
                    download_formats=['svg', 'png', 'jpg', 'pdf', 'eps', 'webp']
                )
                new_output = Output(
                    output_id=output_id,
                    output_type='graph',
                    output_data=output_data
                )
                project.outputs.append(new_output)
                updated_project_content = project.to_json()
                compressed_proj_content = compression.compress(updated_project_content)
                database.write_file(project_id, compressed_proj_content)
                return render_block(
                    environment=environment,
                    template_name="editor/editor.html",
                    block_name="outputs",
                    outputs_data=project.outputs,
                    project_id=project_id
                )
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
                metric = request.args.get("unmixMetric", "cross_correlation")
                output_types = request.args.getlist("outputType")
                sample_names = request.args.getlist("sampleNames")
                spreadsheet_data = spreadsheet.text_to_array(project.data)
                loaded_samples = data.read_1d_samples(spreadsheet_data)
                active_samples = []
                for sample in loaded_samples:
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                adjusted_samples = []
                for sample in active_samples:
                    if project.settings.matrix_function_type == "kde":
                        sample.replace_grain_uncertainties(project.settings.kde_bandwidth)
                    adjusted_samples.append(sample)
                x_min = data.get_x_min(active_samples)
                x_max = data.get_x_max(active_samples)
                sample_pdps = [univariate.distributions.pdp_function(sample, x_min, x_max) for sample in adjusted_samples]
                if metric == 'cross_correlation':
                    sink_y_values = sample_pdps[0].y_values
                    sources_y_values = [sample_pdp.y_values for sample_pdp in sample_pdps[1:]]
                else:
                    sink_y_values = univariate.distributions.cdf_function(sample_pdps[0]).y_values
                    sources_y_values = [univariate.distributions.cdf_function(sample_pdp).y_values for sample_pdp in sample_pdps[1:]]
                contributions, stdevs, top_lines = (
                    univariate.unmix.monte_carlo_model(
                        sink_y_values=sink_y_values,
                        sources_y_values=sources_y_values,
                        n_trials=int(project.settings.n_unmix_trials),
                        metric=metric
                    )
                )
                contribution_pairs = []
                for i, sample in enumerate(active_samples[1:]):
                    contribution_pairs.append(
                        unmix.Contribution(
                            name = sample.name,
                            contribution = contributions[i],
                            standard_deviation=stdevs[i]
                        )
                    )
                if "contribution_table" in output_types:
                    matrix_df = univariate.unmix.relative_contribution_table(
                        contributions=contribution_pairs,
                        metric=metric
                    )
                    output_id = secrets.token_hex(15)
                    output_data = embedding.embed_matrix(
                        dataframe=matrix_df,
                        output_id=output_id,
                        project_id=project_id,
                        download_formats=['xlsx', 'xls', 'csv']
                    )
                    project.outputs.append(
                        Output(
                            output_id=output_id,
                            output_type='matrix',
                            output_data=output_data
                        )
                    )
                if "contribution_graph" in output_types:
                    graph_fig = univariate.unmix.relative_contribution_graph(
                        contributions=contribution_pairs,
                        title=f"{output_title} (metric='{metric}')",
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
                        font_size=project.settings.font_size,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height
                    )
                    output_id = secrets.token_hex(15)
                    output_data = embedding.embed_graph(
                        fig=graph_fig,
                        output_id=output_id,
                        project_id=project_id,
                        fig_type="matplotlib",
                        img_format='svg',
                        download_formats=['svg', 'png', 'jpg', 'pdf', 'eps', 'webp']
                    )
                    project.outputs.append(
                        Output(
                            output_id=output_id,
                            output_type='graph',
                            output_data=output_data
                        )
                    )
                if "trials_graph" in output_types:
                    graph_fig = univariate.unmix.top_trials_graph(
                        sink_line=sink_y_values,
                        model_lines=top_lines,
                        x_range=[x_min, x_max],
                        title=f"{output_title} (metric='{metric}')",
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
                        font_size=project.settings.font_size,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height
                    )
                    output_id = secrets.token_hex(15)
                    output_data = embedding.embed_graph(
                        fig=graph_fig,
                        output_id=output_id,
                        project_id=project_id,
                        fig_type="matplotlib",
                        img_format='svg',
                        download_formats=['svg', 'png', 'jpg', 'pdf', 'eps', 'webp']
                    )
                    project.outputs.append(
                        Output(
                            output_id=output_id,
                            output_type='graph',
                            output_data=output_data
                        )
                    )
                updated_project_content = project.to_json()
                compressed_proj_content = compression.compress(updated_project_content)
                database.write_file(project_id, compressed_proj_content)
                return render_block(
                    environment=environment,
                    template_name="editor/editor.html",
                    block_name="outputs",
                    outputs_data=project.outputs,
                    project_id=project_id
                )
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
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                adjusted_samples = []
                for sample in active_samples:
                    if project.settings.matrix_function_type == "kde":
                        sample.replace_grain_uncertainties(project.settings.kde_bandwidth)
                    adjusted_samples.append(sample)
                adjusted_samples.reverse()
                matrix_df = matrices.generate_data_frame(
                    samples=adjusted_samples,
                    metric=output_type
                )
                output_id = secrets.token_hex(15)
                output_data = embedding.embed_matrix(
                    dataframe=matrix_df,
                    output_id=output_id,
                    title=output_title,
                    project_id=project_id,
                    download_formats=['xlsx', 'xls', 'csv'],
                )
                new_output = Output(
                    output_id=output_id,
                    output_type='matrix',
                    output_data=output_data
                )
                project.outputs.append(new_output)
                updated_project_content = project.to_json()
                compressed_proj_content = compression.compress(updated_project_content)
                database.write_file(project_id, compressed_proj_content)
                return render_block(
                    environment=environment,
                    template_name="editor/editor.html",
                    block_name="outputs",
                    outputs_data=project.outputs,
                    project_id=project_id
                )
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
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                bivariate_distro = kde_function_2d(active_samples[0])
                if output_type == 'kde_2d_surface':
                    fig_type = "plotly"
                    graph_fig = kde_graph_2d(
                        bivariate_distro=bivariate_distro,
                        title=output_title,
                        font_name=project.settings.font_name,
                        font_size=project.settings.font_size,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height
                    )
                    img_format='png'
                elif output_type == 'kde_2d_heatmap':
                    fig_type = "matplotlib"
                    graph_fig = heatmap(
                        bivariate_distro=bivariate_distro,
                        show_points=True,
                        title=output_title,
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
                        font_size=project.settings.font_size,
                        color_map=project.settings.color_map,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height
                    )
                    img_format='png'
                else:
                    raise ValueError("output_type is not supported")
                output_id = secrets.token_hex(15)
                output_data = embedding.embed_graph(
                    fig=graph_fig,
                    output_id=output_id,
                    project_id=project_id,
                    fig_type=fig_type,
                    img_format=img_format,
                    download_formats=['svg', 'png', 'jpg', 'pdf', 'webp']
                )
                new_output = Output(
                    output_id=output_id,
                    output_type='graph',
                    output_data=output_data
                )
                project.outputs.append(new_output)
                updated_project_content = project.to_json()
                compressed_proj_content = compression.compress(updated_project_content)
                database.write_file(project_id, compressed_proj_content)
                return render_block(
                    environment=environment,
                    template_name="editor/editor.html",
                    block_name="outputs",
                    outputs_data=project.outputs,
                    project_id=project_id
                )
            else:
                return jsonify({"outputs": "method not allowed"})
        else:
            return jsonify({"outputs": "access_denied"})


    @app.route('/projects/<int:project_id>/outputs/clear', methods=['POST'])
    @login_required
    def clear_outputs(project_id):
        if session.get("open_project", 0) == project_id:
            file = database.get_file(project_id)
            project_content = compression.decompress(file.content)
            project = project_from_json(project_content)
            project.outputs = []
            updated_project_content = project.to_json()
            compressed_proj_content = compression.compress(updated_project_content)
            database.write_file(project_id, compressed_proj_content)
            project_outputs = project.outputs
            return render_block(environment=environment,
                                template_name="editor/editor.html",
                                block_name="outputs",
                                outputs_data=project_outputs,
                                project_id=project_id)
        else:
            return jsonify({"outputs": "access_denied"})

def is_float(element: any) -> bool:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def __get_project(project_id):
    if session.get("open_project", 0) == project_id:
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        return project_from_json(project_content)
    else:
        return None
