import dz_lib.utils.encode
from flask import render_template, request, jsonify, session
from flask_login import login_required, current_user
from server import database
from utils import spreadsheet, compression
from utils.output import Output
from utils.project import project_from_json
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2_fragments import render_block
import zlib
import secrets
import base64
from dz_lib import univariate, bivariate
from dz_lib.bivariate.distributions import *
from dz_lib.univariate import mds, unmix, distributions, mda
from dz_lib.utils import data, matrices
from utils import embedding, monte_carlo_optimized
from flask import send_file
from pathvalidate import sanitize_filename
import pandas as pd
import json
import uuid
import os
import datetime
import base64
from werkzeug.utils import secure_filename

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
                                   project_id=project_id,
                                   project_name=project.name)
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

    @app.route('/projects/<int:project_id>/data/export', methods=['GET'])
    @login_required
    def export_data(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            spreadsheet_data = spreadsheet.text_to_array(project.data)
            filename = sanitize_filename(request.args.get("filename", "exported_data"))
            file_format = request.args.get("format", "xlsx")
            try:
                df = pd.DataFrame(spreadsheet_data)
                if file_format == "xlsx":
                    buffer = matrices.to_xlsx(df, include_header=False, include_index=False)
                    return send_file(
                        buffer,
                        as_attachment=True,
                        download_name=f"{filename}.xlsx",
                        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif file_format == "csv":
                    buffer = matrices.to_csv(df, include_header=False, include_index=False)
                    buffer.seek(0)
                    return send_file(
                        buffer,
                        as_attachment=True,
                        download_name=f"{filename}.csv",
                        mimetype="text/csv"
                    )
                elif file_format == "xls":
                    buffer = matrices.to_xls(df, include_header=False, include_index=False)
                    buffer.seek(0)
                    return send_file(
                        buffer,
                        as_attachment=True,
                        download_name=f"{filename}.csv",
                        mimetype=dz_lib.utils.encode.get_mime_type("xls")
                    )
                elif file_format == "json":
                    json_data = df.to_json(orient='values')
                    return jsonify(json_data)
                return jsonify({"success": False, "error": "Unsupported format"})
            except Exception as e:
                print(f"Error exporting data: {e}")
                return jsonify({"success": False, "error": str(e)})
        else:
            return jsonify({"error": "access_denied"})

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
                    sample.name = clean_sample_name(sample.name)
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                adjusted_samples = []
                for sample in active_samples:
                    if project.settings.matrix_function_type == "kde" and output_type != "pdp":
                        sample.replace_grain_uncertainties(project.settings.kde_bandwidth)
                    adjusted_samples.append(sample)
                adjusted_samples.reverse()

                if output_type == 'kde':
                    distros = []
                    for sample in adjusted_samples:
                        distros.append(
                            univariate.distributions.kde_function(
                                sample=sample,
                                bandwidth=float(project.settings.kde_bandwidth)
                            )
                        )
                    graph_fig = univariate.distributions.distribution_graph(
                        distributions=distros,
                        stacked=project.settings.stack_graphs == "true",
                        legend=project.settings.legend == "true",
                        title=output_title,
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
                        font_size=project.settings.font_size,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height,
                        color_map=project.settings.color_map,
                        x_min=project.settings.min_age,
                        x_max=project.settings.max_age
                    )
                elif output_type == 'pdp':
                    distros = []
                    for sample in adjusted_samples:
                        distros.append(
                            univariate.distributions.pdp_function(sample)
                        )
                    graph_fig = univariate.distributions.distribution_graph(
                        distributions=distros,
                        stacked=project.settings.stack_graphs == "true",
                        legend=project.settings.legend == "true",
                        title=output_title,
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
                        font_size=project.settings.font_size,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height,
                        color_map=project.settings.color_map,
                        x_min=project.settings.min_age,
                        x_max=project.settings.max_age
                    )
                elif output_type == 'cdf':
                    distros = []
                    for sample in adjusted_samples:
                        distros.append(
                            univariate.distributions.cdf_function(
                                univariate.distributions.kde_function(
                                    sample=sample,
                                    bandwidth=float(project.settings.kde_bandwidth)
                                )
                            )
                        )
                    graph_fig = univariate.distributions.distribution_graph(
                        distributions=distros,
                        stacked=project.settings.stack_graphs == "true",
                        legend=project.settings.legend == "true",
                        title=output_title,
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
                        font_size=project.settings.font_size,
                        fig_width=project.settings.figure_width,
                        fig_height=project.settings.figure_height,
                        color_map=project.settings.color_map,
                        x_min=project.settings.min_age,
                        x_max=project.settings.max_age
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
                    download_formats=['svg', 'png', 'jpg', 'pdf', 'eps']
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
                    sample.name = clean_sample_name(sample.name)
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
                    download_formats=['svg', 'png', 'jpg', 'pdf', 'eps']
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
                    sample.name = clean_sample_name(sample.name)
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
                    monte_carlo_optimized.monte_carlo_model_optimized(
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
                        download_formats=['svg', 'png', 'jpg', 'pdf', 'eps']
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
                        download_formats=['svg', 'png', 'jpg', 'pdf', 'eps']
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
                    sample.name = clean_sample_name(sample.name)
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
                    sample.name = clean_sample_name(sample.name)
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                bivariate_distro = kde_function_2d(active_samples[0])
                if output_type == 'kde_2d_surface':
                    fig_type = "plotly"
                    graph_fig = kde_graph_2d(
                        bivariate_distro=bivariate_distro,
                        title=output_title,
                        font_path=f'static/global/fonts/{project.settings.font_name}.ttf',
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
                    download_formats=['svg', 'png', 'jpg', 'pdf']
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
                    sample.name = clean_sample_name(sample.name)
                    for sample_name in sample_names:
                        if sample.name == sample_name:
                            active_samples.append(sample)
                sample = active_samples[0]
                if "mda_table" in output_types:
                    matrix_df = univariate.mda.comparison_table(sample.grains)
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
                if "mda_graph" in output_types:
                    graph_fig = univariate.mda.comparison_graph(
                        grains=sample.grains,
                        title=output_title,
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
                        download_formats=['svg', 'png', 'jpg', 'pdf', 'eps']
                    )
                    project.outputs.append(
                        Output(
                            output_id=output_id,
                            output_type='graph',
                            output_data=output_data
                        )
                    )
                if "rank_plot" in output_types:
                    graph_fig = univariate.mda.ranked_ages_plot(
                        grains=sample.grains,
                        title=output_title,
                        x_min=project.settings.min_age,
                        x_max=project.settings.max_age,
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
                        download_formats=['svg', 'png', 'jpg', 'pdf', 'eps']
                    )
                    project.outputs.append(
                        Output(
                            output_id=output_id,
                            output_type='graph',
                            output_data=output_data
                        )
                    )
                if "ygf_graph" in output_types:
                    distro = distributions.pdp_function(sample)
                    fitted_grain, fitted_distro = mda.youngest_gaussian_fit(sample.grains)
                    graph_fig = distributions.distribution_graph(
                        distributions=[distro, fitted_distro],
                        color_map="rainbow",
                        x_min=project.settings.min_age,
                        x_max=project.settings.max_age,
                        title=output_title,
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
                        download_formats=['svg', 'png', 'jpg', 'pdf', 'eps']
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

    @app.route('/projects/<int:project_id>/outputs', methods=['GET'])
    @login_required
    def get_project_outputs(project_id):
        if session.get("open_project", 0) == project_id:
            try:
                project = __get_project(project_id)
                outputs_list = []
                
                for output in project.outputs:
                    outputs_list.append({
                        'output_id': output.output_id,
                        'output_type': output.output_type,
                        'output_data': output.output_data
                    })
                
                return jsonify(outputs_list)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    # Maps functionality routes
    @app.route('/projects/<int:project_id>/maps/points', methods=['GET'])
    @login_required
    def get_map_points(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            # Get map points from project settings
            map_points = getattr(project.settings, 'map_points', [])
            return jsonify(map_points)
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/maps/points', methods=['POST'])
    @login_required
    def add_map_point(project_id):
        if session.get("open_project", 0) == project_id:
            try:
                project = __get_project(project_id)
                data = request.get_json()

                # Validate required fields
                required_fields = ['latitude', 'longitude', 'title']
                for field in required_fields:
                    if field not in data or not data[field]:
                        return jsonify({'error': f'Missing required field: {field}'}), 400

                # Create new point
                point = {
                    'id': str(uuid.uuid4()),
                    'latitude': float(data['latitude']),
                    'longitude': float(data['longitude']),
                    'title': data['title'],
                    'image_url': data.get('image_url', '')
                }

                # Get existing map points or initialize empty list
                map_points = getattr(project.settings, 'map_points', [])
                map_points.append(point)
                project.settings.map_points = map_points

                # Save project
                updated_project_content = project.to_json()
                compressed_proj_content = compression.compress(updated_project_content)
                database.write_file(project_id, compressed_proj_content)

                return jsonify(point), 201

            except ValueError:
                return jsonify({'error': 'Invalid latitude or longitude'}), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/maps/points/<point_id>', methods=['DELETE'])
    @login_required
    def delete_map_point(project_id, point_id):
        if session.get("open_project", 0) == project_id:
            try:
                project = __get_project(project_id)
                map_points = getattr(project.settings, 'map_points', [])
                original_length = len(map_points)
                
                # Remove point with matching ID
                map_points = [point for point in map_points if point['id'] != point_id]
                
                if len(map_points) == original_length:
                    return jsonify({'error': 'Point not found'}), 404

                project.settings.map_points = map_points

                # Save project
                updated_project_content = project.to_json()
                compressed_proj_content = compression.compress(updated_project_content)
                database.write_file(project_id, compressed_proj_content)

                return jsonify({'message': 'Point deleted successfully'}), 200

            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/maps/upload', methods=['POST'])
    @login_required
    def upload_map_image(project_id):
        if session.get("open_project", 0) == project_id:
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400

                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400

                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filename = f"{uuid.uuid4()}_{filename}"
                    
                    # Create uploads directory if it doesn't exist
                    uploads_dir = os.path.join('static', 'uploads')
                    os.makedirs(uploads_dir, exist_ok=True)
                    
                    filepath = os.path.join(uploads_dir, filename)
                    file.save(filepath)

                    file_url = f"/static/uploads/{filename}"
                    return jsonify({'url': file_url}), 200
                else:
                    return jsonify({'error': 'Invalid file type'}), 400

            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/maps/clear', methods=['POST'])
    @login_required
    def clear_map_points(project_id):
        if session.get("open_project", 0) == project_id:
            try:
                project = __get_project(project_id)
                project.settings.map_points = []

                # Save project
                updated_project_content = project.to_json()
                compressed_proj_content = compression.compress(updated_project_content)
                database.write_file(project_id, compressed_proj_content)

                return jsonify({'message': 'All points cleared successfully'}), 200

            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/maps/export', methods=['POST'])
    @login_required
    def export_map(project_id):
        if session.get("open_project", 0) == project_id:
            try:
                data = request.get_json()
                map_data = data.get('map_data')

                if not map_data:
                    return jsonify({'error': 'No map data provided'}), 400

                html_content = generate_static_map_html(map_data)

                export_filename = f"static_map_{uuid.uuid4()}.html"
                export_path = os.path.join('temp')
                os.makedirs(export_path, exist_ok=True)
                full_path = os.path.join(export_path, export_filename)

                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                return send_file(full_path, as_attachment=True, download_name='static_map.html')

            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_static_map_html(map_data):
    points = map_data.get('points', [])
    center = map_data.get('center', {'lat': 40.7128, 'lng': -74.0060})
    zoom = map_data.get('zoom', 10)
    settings = map_data.get('settings', {})

    map_title = settings.get('title', 'Interactive Map Export')
    map_subtitle = settings.get('subtitle', '')
    map_footer = settings.get('footer', 'This is a static export of your interactive map.')

    markers_js = ""
    for point in points:
        popup_content = '<div class="point-info">'
        if point.get('image_url'):
            try:
                if point['image_url'].startswith('data:image'):
                    # Already a data URL, use directly
                    popup_content += f'<img src="{point["image_url"]}" alt="{point["title"]}" class="popup-image" style="max-width:100%; max-height:150px; cursor:pointer;" onclick="enlargeImage(this)">'
                elif point['image_url'].startswith('/static/uploads/'):
                    # File upload, convert to data URL
                    file_path = point['image_url'].replace('/static/', 'static/')
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as img_file:
                            img_data = img_file.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            ext = file_path.lower().split('.')[-1]
                            mime_type_map = {
                                'jpg': 'image/jpeg',
                                'jpeg': 'image/jpeg',
                                'png': 'image/png',
                                'gif': 'image/gif',
                                'webp': 'image/webp',
                                'svg': 'image/svg+xml'
                            }
                            mime_type = mime_type_map.get(ext, 'image/jpeg')
                            data_uri = f'data:{mime_type};base64,{img_base64}'
                            popup_content += f'<img src="{data_uri}" alt="{point["title"]}" class="popup-image" style="max-width:100%; max-height:150px; cursor:pointer;" onclick="enlargeImage(this)">'
                else:
                    # Regular URL, use as-is
                    popup_content += f'<img src="{point["image_url"]}" alt="{point["title"]}" class="popup-image" style="max-width:100%; max-height:150px; cursor:pointer;" onclick="enlargeImage(this)">'
            except Exception as e:
                print(f"Error encoding image for point {point.get('title', 'Unknown')}: {e}")

        popup_content += f'<h6>{point["title"]}</h6>'
        popup_content += f'<small>Lat: {point["latitude"]}, Lng: {point["longitude"]}</small>'
        popup_content += '</div>'
        popup_content_escaped = popup_content.replace('`', '\\`').replace('${', '\\${')

        markers_js += f"""
        L.marker([{point['latitude']}, {point['longitude']}])
            .addTo(map)
            .bindPopup(`{popup_content_escaped}`);
        """

    export_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    html_template = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{map_title}</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css" rel="stylesheet">
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
        #map {{ height: 600px; width: 100%; }}
        .header {{ text-align: center; background: #007bff; color: white; padding: 1rem; }}
        .footer {{ text-align: center; padding: 1rem; background: #f8f9fa; }}

        /* Image Modal Styles */
        .image-modal {{
            display: none;
            position: fixed;
            z-index: 10000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            backdrop-filter: blur(5px);
        }}

        .image-modal.show {{
            display: flex;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.3s ease-out;
        }}

        .modal-content {{
            position: relative;
            max-width: 90vw;
            max-height: 90vh;
            margin: auto;
        }}

        .enlarged-image {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }}

        .close-button {{
            position: absolute;
            top: -40px;
            right: 0;
            color: white;
            font-size: 35px;
            font-weight: bold;
            cursor: pointer;
            background: rgba(0, 0, 0, 0.5);
            border: none;
            width: 45px;
            height: 45px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }}

        .close-button:hover {{
            background: rgba(0, 0, 0, 0.8);
            transform: scale(1.1);
        }}

        .popup-image {{
            transition: opacity 0.2s ease;
        }}

        .popup-image:hover {{
            opacity: 0.8;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        /* Mobile responsiveness */
        @media (max-width: 768px) {{
            .close-button {{
                top: 10px;
                right: 10px;
                font-size: 24px;
                width: 35px;
                height: 35px;
            }}
        }}
    </style>
</head>
<body>
    <!-- Image Modal -->
    <div id="imageModal" class="image-modal" onclick="closeImageModal(event)">
        <div class="modal-content">
            <button class="close-button" onclick="closeImageModal()">&times;</button>
            <img id="enlargedImage" class="enlarged-image" src="" alt="">
        </div>
    </div>

    <div class="header">
        <h1>{map_title}</h1>
        {f'<p>{map_subtitle}</p>' if map_subtitle else ''}
        <small>Exported on {export_id} • {len(points)} point(s)</small>
    </div>
    <div id="map"></div>
    <div class="footer">{map_footer}</div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
    <script>
        const map = L.map('map').setView([{center['lat']}, {center['lng']}], {zoom});
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ attribution: '© OpenStreetMap contributors' }}).addTo(map);
        {markers_js}

        // Image enlargement functionality
        function enlargeImage(img) {{
            const modal = document.getElementById('imageModal');
            const enlargedImg = document.getElementById('enlargedImage');

            enlargedImg.src = img.src;
            enlargedImg.alt = img.alt;
            modal.classList.add('show');

            // Prevent event bubbling
            event.stopPropagation();
        }}

        function closeImageModal(event) {{
            const modal = document.getElementById('imageModal');

            // Only close if clicking on the modal backdrop or close button
            if (!event || event.target === modal || event.target.classList.contains('close-button')) {{
                modal.classList.remove('show');
            }}
        }}

        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeImageModal();
            }}
        }});

        // Prevent modal from closing when clicking on the image itself
        document.getElementById('enlargedImage').addEventListener('click', function(event) {{
            event.stopPropagation();
        }});
    </script>
</body>
</html>'''

    return html_template

def is_float(element: any) -> bool:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def clean_sample_name(sample_name):
    try:
        num = float(sample_name)
        if num.is_integer():
            return str(int(num))
    except ValueError:
        pass
    return str(sample_name)

def __get_project(project_id):
    if session.get("open_project", 0) == project_id:
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        return project_from_json(project_content)
    else:
        return None
