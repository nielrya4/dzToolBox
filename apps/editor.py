from flask import render_template, request, jsonify, session
from flask_login import login_required, current_user
from server import database
from jinja2 import Environment, FileSystemLoader, select_autoescape
from utils import spreadsheet, unmix, compression
from utils.output import Output
import json
import zlib
import secrets
from utils.graph import Graph
from utils.matrix import Matrix
from utils.project import Project
from jinja2_fragments import render_block
import base64


environment = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(("html", "jinja2"))
)


def register(app):
    @app.route('/projects/<int:project_id>', methods=['GET', 'POST'])
    @login_required
    def open_project(project_id):
        session["open_project"] = project_id
        file = database.get_file(project_id)
        project_content = file.content
        decompressed_project = compression.decompress(project_content)
        project_author = f"<User {file.user_id}>"
        if str(project_author) == str(current_user):
            project_data = get_project_data(decompressed_project)
            spreadsheet_data = json.loads(project_data)
            loaded_samples = spreadsheet.read_samples(spreadsheet_data)

            samples_data = []
            for sample in loaded_samples:
                active = request.form.get(sample.name) == "true"
                samples_data.append([sample.name, active])

            project_outputs = get_all_outputs(decompressed_project)
            if not project_outputs:
                project_outputs = [Output("Default", "graph", "<h1>No Outputs Yet</h1>")]
            return render_template("editor/editor.html",
                                   spreadsheet_data=project_data,
                                   samples=samples_data,
                                   outputs_data=project_outputs)
        else:
            return render_template("errors/403.html")

    @app.route('/json/save/spreadsheet', methods=['POST'])
    @login_required
    def json_save_spreadsheet():
        if session.get("open_project", 0) != 0:
            try:
                compressed_data = request.get_json().get('compressedData', '')
                if not compressed_data:
                    return jsonify({"success": False, "error": "No compressed data provided"})
                compressed_data_bytes = base64.b64decode(compressed_data)
                decompressed_data = zlib.decompress(compressed_data_bytes).decode('utf-8')
                json_data = json.loads(decompressed_data)
                project_id = session["open_project"]
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
                file = database.get_file(project_id)
                project_content = compression.decompress(file.content)
                project_json = json.loads(project_content)
                project_json["data"] = spreadsheet.array_to_text(data)
                updated_project_content = json.dumps(project_json)
                compressed_proj_content = compression.compress(updated_project_content)
                database.write_file(project_id, compressed_proj_content)
                return jsonify({"success": True})
            except Exception as e:
                print(f"Error updating project data: {e}")
                return jsonify({"success": False, "error": str(e)})

        return jsonify({"result": "ok", "filename": "filename"})

    @app.route('/get_sample_names', methods=['POST'])
    @login_required
    def get_sample_names():
        if session.get("open_project", 0) != 0:
            project_id = session.get("open_project", 0)
            file = database.get_file(project_id)
            project_content = compression.decompress(file.content)
            project_data = get_project_data(project_content)
            spreadsheet_data = spreadsheet.text_to_array(project_data)
            try:
                samples = spreadsheet.read_samples(spreadsheet_data)
                sample_names = [sample.name for sample in samples]
                return jsonify({"sample_names": sample_names})
            except Exception as e:
                print(f"Error updating project data: {e}")
                return jsonify({"success": False, "error": str(e)})
        return jsonify({"sample_names": "failed"})

    @app.route('/update_settings', methods=['POST'])
    @login_required
    def update_settings():
        project_id = session.get("open_project", 0)
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        settings = request.get_json()
        updated_project_content = set_project_settings(project_content, settings)
        compressed_proj_content = compression.compress(updated_project_content)
        database.write_file(project_id, compressed_proj_content)
        return jsonify({"result": "ok"})

    @app.route('/get_settings', methods=['POST'])
    @login_required
    def get_settings():
        project_id = session.get("open_project", 0)
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        settings = get_project_settings(project_content)
        return jsonify({"settings" : settings})

    @app.route('/delete_output/<string:output_id>', methods=['POST'])
    @login_required
    def del_output(output_id):
        project_id = session.get("open_project", 0)
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        if get_all_outputs(project_content) is None:
            outputs = []
        else:
            outputs = get_all_outputs(project_content)
        for output in outputs:
            if output.id == output_id:
                outputs.remove(output)
        updated_project_content = set_all_outputs(project_content, outputs)
        compressed_proj_content = compression.compress(updated_project_content)
        database.write_file(project_id, compressed_proj_content)
        project_outputs = get_all_outputs(updated_project_content)
        return render_block(environment=environment,
                            template_name="editor/editor.html",
                            block_name="outputs",
                            outputs_data=project_outputs)


    @app.route('/clear_outputs', methods=['POST'])
    @login_required
    def clear_outputs():
        project_id = session.get("open_project", 0)
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        if get_all_outputs(project_content) is None:
            outputs = []
        else:
            outputs = get_all_outputs(project_content)

        outputs.clear()

        updated_project_content = set_all_outputs(project_content, outputs)
        compressed_proj_content = compression.compress(updated_project_content)
        database.write_file(project_id, compressed_proj_content)
        project_outputs = get_all_outputs(updated_project_content)
        return render_block(environment=environment,
                            template_name="editor/editor.html",
                            block_name="outputs",
                            outputs_data=project_outputs)

    @app.route('/new_distro', methods=['GET'])
    @login_required
    def new_distro():
        output_name = request.args.get('output_name', '')
        sample_names = request.args.getlist('samples')
        output_type = request.args.get('distro_type', '')
        output_id = secrets.token_hex(15)
        project_id = session.get("open_project", 0)
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        project_data = get_project_data(project_content)
        project_settings = get_project_settings(project_content)
        spreadsheet_data = spreadsheet.text_to_array(project_data)
        loaded_samples = spreadsheet.read_samples(spreadsheet_data)
        kde_bandwidth_setting = project_settings["kde_bandwidth"] if project_settings["kde_bandwidth"] is not None else 10
        actions_button_setting = project_settings["actions_button"] == "true" if project_settings["actions_button"] is not None else False
        stack_graphs_setting = project_settings["stack_graphs"] == "true" if project_settings["stack_graphs"] is not None else False
        matrix_function_type_setting = project_settings["matrix_function_type"] if project_settings["matrix_function_type"] is not None else "kde"
        color_map = project_settings["graph_figure_settings"]["graph_color_map"] if project_settings["graph_figure_settings"]["graph_color_map"] is not None else "plasma"
        font_size = project_settings["graph_figure_settings"]["font_size"] if project_settings["graph_figure_settings"]["font_size"] is not None else 12
        font_name = project_settings["graph_figure_settings"]["font_name"] if project_settings["graph_figure_settings"]["font_name"] is not None else "ubuntu"
        figure_width = project_settings["graph_figure_settings"]["figure_width"] if project_settings["graph_figure_settings"]["figure_width"] is not None else 9
        figure_height = project_settings["graph_figure_settings"]["figure_height"] if project_settings["graph_figure_settings"]["figure_height"] is not None else 7
        min_age = project_settings["min_age"] if project_settings["min_age"] is not None else 0
        max_age = project_settings["max_age"] if project_settings["max_age"] is not None else 4500
        output_data = ""
        active_samples = []
        for sample in loaded_samples:
            for sample_name in sample_names:
                if sample.name == sample_name:
                    active_samples.append(sample)
        adjusted_samples = []
        for sample in active_samples:
            if matrix_function_type_setting == "kde" and output_type != "pdp_graph":
                sample.replace_bandwidth(10)
            adjusted_samples.append(sample)
        adjusted_samples.reverse()
        if output_type == "kde_graph":
            kde_bandwidth = kde_bandwidth_setting
            graph = Graph(samples=adjusted_samples,
                          title=output_name,
                          stacked=stack_graphs_setting,
                          graph_type="kde",
                          kde_bandwidth=kde_bandwidth,
                          color_map=color_map,
                          font_name=font_name,
                          font_size=font_size,
                          fig_width=figure_width,
                          fig_height=figure_height,
                          min_age=min_age,
                          max_age=max_age)
            output_data = graph.generate_html(output_id, actions_button=actions_button_setting)
            output_type = "graph"
        elif output_type == "pdp_graph":
            graph = Graph(samples=adjusted_samples,
                          title=output_name,
                          stacked=stack_graphs_setting,
                          graph_type="pdp",
                          color_map=color_map,
                          font_name=font_name,
                          font_size=font_size,
                          fig_width=figure_width,
                          fig_height=figure_height,
                          min_age=min_age,
                          max_age=max_age)
            output_data = graph.generate_html(output_id, actions_button=actions_button_setting)
            output_type = "graph"
        elif output_type == "cdf_graph":
            graph = Graph(samples=adjusted_samples,
                          title=output_name,
                          stacked=stack_graphs_setting,
                          graph_type="cdf",
                          color_map=color_map,
                          font_name=font_name,
                          font_size=font_size,
                          fig_width=figure_width,
                          fig_height=figure_height,
                          min_age=min_age,
                          max_age=max_age)
            output_data = graph.generate_html(output_id, actions_button=actions_button_setting)
            output_type = "graph"
        if get_all_outputs(project_content) is None:
            outputs = []
        else:
            outputs = get_all_outputs(project_content)
        output = Output(output_id, output_type, output_data)
        outputs.append(output)
        updated_project_content = set_all_outputs(project_content, outputs)
        compressed_proj_content = compression.compress(updated_project_content)
        database.write_file(project_id, compressed_proj_content)
        project_outputs = get_all_outputs(updated_project_content)
        return render_block(environment=environment,
                            template_name="editor/editor.html",
                            block_name="outputs",
                            outputs_data=project_outputs)

    @app.route('/new_matrix', methods=['GET'])
    @login_required
    def new_matrix():
        output_name = request.args.get('output_name', '')
        sample_names = request.args.getlist('samples')
        matrix_type = request.args.get('matrix_type', '')
        output_id = secrets.token_hex(15)
        project_id = session.get("open_project", 0)
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        project_data = get_project_data(project_content)
        project_settings = get_project_settings(project_content)
        spreadsheet_data = spreadsheet.text_to_array(project_data)
        loaded_samples = spreadsheet.read_samples(spreadsheet_data)
        kde_bandwidth_setting = project_settings["kde_bandwidth"] if project_settings["kde_bandwidth"] is not None else 10
        actions_button_setting = project_settings["actions_button"] == "true" if project_settings["actions_button"] is not None else False
        matrix_function_type_setting = project_settings["matrix_function_type"] if project_settings["matrix_function_type"] is not None else "kde"
        output_data = ""
        active_samples = []
        for sample in loaded_samples:
            for sample_name in sample_names:
                if sample.name == sample_name:
                    active_samples.append(sample)
        adjusted_samples = []
        for sample in active_samples:
            if matrix_function_type_setting == "kde":
                sample.replace_bandwidth(kde_bandwidth_setting)
            adjusted_samples.append(sample)
        adjusted_samples.reverse()
        if matrix_type == "similarity":
            matrix = Matrix(adjusted_samples, "similarity", function_type=matrix_function_type_setting)
            output_data = matrix.to_html(output_id, actions_button=actions_button_setting)
            output_type = "matrix"
        elif matrix_type == "likeness":
            matrix = Matrix(adjusted_samples, "likeness", function_type=matrix_function_type_setting)
            output_data = matrix.to_html(output_id, actions_button=actions_button_setting)
            output_type = "matrix"
        elif matrix_type == "ks_matrix":
            matrix = Matrix(active_samples, "ks")
            output_data = matrix.to_html(output_id, actions_button=actions_button_setting)
            output_type = "matrix"
        elif matrix_type == "kuiper":
            matrix = Matrix(active_samples, "kuiper")
            output_data = matrix.to_html(output_id, actions_button=actions_button_setting)
            output_type = "matrix"
        elif matrix_type == "cross_correlation":
            matrix = Matrix(adjusted_samples, "r2", function_type=matrix_function_type_setting)
            output_data = matrix.to_html(output_id, actions_button=actions_button_setting)
            output_type = "matrix"
        elif matrix_type == "dis_similarity":
            matrix = Matrix(adjusted_samples, "dissimilarity", function_type=matrix_function_type_setting)
            output_data = matrix.to_html(output_id, actions_button=actions_button_setting)
            output_type = "matrix"
        elif matrix_type == "dis_likeness":
            matrix = Matrix(adjusted_samples, "similarity", function_type=matrix_function_type_setting)
            output_data = matrix.to_html(output_id, actions_button=actions_button_setting)
            output_type = "matrix"
        elif matrix_type == "dis_ks":
            matrix = Matrix(active_samples, "similarity", function_type=matrix_function_type_setting)
            output_data = matrix.to_html(output_id, actions_button=actions_button_setting)
            output_type = "matrix"
        elif matrix_type == "dis_kuiper":
            matrix = Matrix(active_samples, "similarity", function_type=matrix_function_type_setting)
            output_data = matrix.to_html(output_id, actions_button=actions_button_setting)
            output_type = "matrix"
        elif matrix_type == "dis_cross_correlation":
            matrix = Matrix(adjusted_samples, "similarity", function_type=matrix_function_type_setting)
            output_data = matrix.to_html(output_id, actions_button=actions_button_setting)
            output_type = "matrix"
        if get_all_outputs(project_content) is None:
            outputs = []
        else:
            outputs = get_all_outputs(project_content)
        output = Output(output_id, output_type, output_data)
        outputs.append(output)
        updated_project_content = set_all_outputs(project_content, outputs)
        compressed_proj_content = compression.compress(updated_project_content)
        database.write_file(project_id, compressed_proj_content)
        project_outputs = get_all_outputs(updated_project_content)
        return render_block(environment=environment,
                            template_name="editor/editor.html",
                            block_name="outputs",
                            outputs_data=project_outputs)

    @app.route('/new_mds', methods=['GET'])
    @login_required
    def new_mds():
        output_name = request.args.get('output_name', '')
        sample_names = request.args.getlist('samples')
        mds_type = request.args.get('mds_type', '')
        output_id = secrets.token_hex(15)
        project_id = session.get("open_project", 0)
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        project_data = get_project_data(project_content)
        project_settings = get_project_settings(project_content)
        spreadsheet_data = spreadsheet.text_to_array(project_data)
        loaded_samples = spreadsheet.read_samples(spreadsheet_data)
        kde_bandwidth_setting = project_settings["kde_bandwidth"] if project_settings["kde_bandwidth"] is not None else 10
        actions_button_setting = project_settings["actions_button"] == "true" if project_settings["actions_button"] is not None else False
        matrix_function_type_setting = project_settings["matrix_function_type"] if project_settings["kde_bandwidth"] is not None else "kde"
        color_map = project_settings["graph_figure_settings"]["graph_color_map"] if project_settings["graph_figure_settings"]["graph_color_map"] is not None else "plasma"
        output_data = ""
        active_samples = []
        for sample in loaded_samples:
            for sample_name in sample_names:
                if sample.name == sample_name:
                    active_samples.append(sample)

        if mds_type == "similarity":
            graph = Graph(samples=active_samples,
                          title=output_name,
                          stacked=False,
                          graph_type="sim_mds",
                          kde_bandwidth=kde_bandwidth_setting,
                          color_map=color_map)
            output_data = graph.generate_html(output_id, actions_button=actions_button_setting)
        elif mds_type == "likeness":
            graph = Graph(samples=active_samples,
                          title=output_name,
                          stacked=False,
                          graph_type="like_mds",
                          kde_bandwidth=kde_bandwidth_setting,
                          color_map=color_map)
            output_data = graph.generate_html(output_id, actions_button=actions_button_setting)
        elif mds_type == "ks":
            graph = Graph(samples=active_samples,
                          title=output_name,
                          stacked=False,
                          graph_type="ks_mds",
                          kde_bandwidth=kde_bandwidth_setting,
                          color_map=color_map)
            output_data = graph.generate_html(output_id, actions_button=actions_button_setting)
        elif mds_type == "kuiper":
            graph = Graph(samples=active_samples,
                          title=output_name,
                          stacked=False,
                          graph_type="kuiper_mds",
                          kde_bandwidth=kde_bandwidth_setting,
                          color_map=color_map)
            output_data = graph.generate_html(output_id, actions_button=actions_button_setting)
        elif mds_type == "cross_correlation":
            graph = Graph(samples=active_samples,
                          title=output_name,
                          stacked=False,
                          graph_type="r2_mds",
                          kde_bandwidth=kde_bandwidth_setting,
                          color_map=color_map)
            output_data = graph.generate_html(output_id, actions_button=actions_button_setting)

        if get_all_outputs(project_content) is None:
            outputs = []
        else:
            outputs = get_all_outputs(project_content)

        output = Output(output_id, 'graph', output_data)
        outputs.append(output)

        updated_project_content = set_all_outputs(project_content, outputs)
        compressed_proj_content = compression.compress(updated_project_content)
        database.write_file(project_id, compressed_proj_content)
        project_outputs = get_all_outputs(updated_project_content)
        return render_block(environment=environment,
                            template_name="editor/editor.html",
                            block_name="outputs",
                            outputs_data=project_outputs)

    @app.route('/new_unmix', methods=['GET'])
    @login_required
    def new_unmix():
        output_name = request.args.get('output_name', '')
        sample_names = request.args.getlist('samples')
        unmix_type = request.args.get('unmix_type', '')
        unmix_outputs = request.args.getlist('unmix_outputs')
        output_ids = [secrets.token_hex(15), secrets.token_hex(15), secrets.token_hex(15)]
        project_id = session.get("open_project", 0)
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        project_data = get_project_data(project_content)
        spreadsheet_data = spreadsheet.text_to_array(project_data)
        loaded_samples = spreadsheet.read_samples(spreadsheet_data)
        active_samples = []
        for sample in loaded_samples:
            for sample_name in sample_names:
                if sample.name == sample_name:
                    active_samples.append(sample)
        contribution_table, contribution_graph, trials_graph = unmix.do_monte_carlo(active_samples, output_ids, num_trials=10000, test_type=unmix_type)
        if get_all_outputs(project_content) is None:
            outputs = []
        else:
            outputs = get_all_outputs(project_content)
        table_output = Output(output_ids[0], 'matrix', contribution_table)
        contribution_graph_output = Output(output_ids[1], 'graph', contribution_graph)
        trials_graph_output = Output(output_ids[2], 'graph', trials_graph)
        if "contribution_table" in unmix_outputs:
            outputs.append(table_output)
        if "contribution_graph" in unmix_outputs:
            outputs.append(contribution_graph_output)
        if "trials_graph" in unmix_outputs:
            outputs.append(trials_graph_output)
        updated_project_content = set_all_outputs(project_content, outputs)
        compressed_proj_content = compression.compress(updated_project_content)
        database.write_file(project_id, compressed_proj_content)
        project_outputs = get_all_outputs(updated_project_content)
        return render_block(environment=environment,
                            template_name="editor/editor.html",
                            block_name="outputs",
                            outputs_data=project_outputs)

    @app.route('/new_hafnium', methods=['GET'])
    @login_required
    def new_hafnium():
        sample_names = request.args.getlist('samples')
        output_name = request.args.get('output_name', '')
        output_type = request.args.get('hafnium_type', '')
        output_id = secrets.token_hex(15)
        project_id = session.get("open_project", 0)
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        project_data = get_project_data(project_content)
        spreadsheet_data = spreadsheet.text_to_array(project_data)
        loaded_samples = spreadsheet.read_samples(spreadsheet_data)
        project_settings = get_project_settings(project_content)
        color_map = project_settings["graph_figure_settings"]["graph_color_map"] if project_settings["graph_figure_settings"]["graph_color_map"] is not None else "plasma"

        active_samples = []
        for sample in loaded_samples:
            for sample_name in sample_names:
                if sample.name == sample_name:
                    active_samples.append(sample)
        if output_type == "density":
            output_content = Graph(title=output_name, samples=active_samples, graph_type="kde2d").generate_fig()
        elif output_type == "heatmap":
            output_content = Graph(title=output_name, samples=active_samples, color_map=color_map, graph_type="heatmap").generate_html(output_id=output_id)

        if get_all_outputs(project_content) is None:
            outputs = []
        else:
            outputs = get_all_outputs(project_content)
        output3d = Output(output_id, 'graph', output_content)
        outputs.append(output3d)

        updated_project_content = set_all_outputs(project_content, outputs)
        compressed_proj_content = compression.compress(updated_project_content)
        database.write_file(project_id, compressed_proj_content)
        project_outputs = get_all_outputs(updated_project_content)
        return render_block(environment=environment,
                            template_name="editor/editor.html",
                            block_name="outputs",
                            outputs_data=project_outputs)


def get_all_outputs(json_string):
    try:
        outputs = []
        data = json.loads(json_string)
        for output in data.get("outputs", []):
            output_id = output.get("output_id")
            output_type = output.get("output_type")
            output_data = output.get("output_data")
            outputs.append(Output(output_id, output_type, output_data))
        return outputs
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None


def set_all_outputs(json_string, outputs):
    name = get_project_name(json_string)
    data = get_project_data(json_string)
    settings = get_project_settings(json_string)
    updated_project = Project(name=name, data=data, outputs=outputs, settings=settings)
    updated_project_content = updated_project.generate_json_string()
    return updated_project_content

def set_project_settings(json_string, settings):
    name = get_project_name(json_string)
    data = get_project_data(json_string)
    outputs = get_all_outputs(json_string)
    updated_project = Project(name=name, data=data, outputs=outputs, settings=settings)
    updated_project_content = updated_project.generate_json_string()
    return updated_project_content


def delete_all_outputs(json_string):
    name = get_project_name(json_string)
    data = get_project_data(json_string)
    updated_project = Project(name=name, data=data, outputs=[])
    updated_project_content = updated_project.generate_json_string()
    return updated_project_content


def delete_output(json_string, output):
    project_outputs = get_all_outputs(json_string)
    for project_output in project_outputs:
        if project_output.name == output.name:
            if project_output.data == output.data:
                project_outputs.remove(project_output)
    updated_project_content = set_all_outputs(json_string, project_outputs)
    return updated_project_content


def get_output_by_name(json_string, output_name):
    try:
        data = json.loads(json_string)
        for output in data.get("outputs", []):
            if output.get("output_name") == output_name:
                return output.get("output_data")
        return None
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None


def get_project_data(json_string):
    try:
        data = json.loads(json_string)
        project_data = data.get("data")
        return project_data
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None

def get_project_settings(json_string):
    try:
        data = json.loads(json_string)
        project_settings = data.get("default_settings")
        return project_settings
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None


def get_project_name(json_string):
    try:
        data = json.loads(json_string)
        project_name = data.get("project_name")
        return project_name
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None


def is_float(element: any) -> bool:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False
