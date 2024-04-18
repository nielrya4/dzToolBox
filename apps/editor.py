from flask import render_template, request, jsonify, session
from flask_login import login_required, current_user
from server import database
from jinja2 import Environment, FileSystemLoader, select_autoescape
from utils import spreadsheet
from utils.output import Output
import json
import openpyxl
import os
from utils.graph import Graph
from utils.matrix import Matrix
from utils.project import Project
from jinja2_fragments import render_block


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

        project_data = get_project_data(project_content)
        spreadsheet_data = spreadsheet.text_to_array(project_data)

        loaded_samples = spreadsheet.read_samples(spreadsheet_data)
        samples_data = []
        for sample in loaded_samples:
            active = request.form.get(sample.name) == "true"
            samples_data.append([sample.name, active])

        project_outputs = get_all_outputs(project_content)
        print(project_outputs)
        if not project_outputs:
            project_outputs = [Output("Default", "graph", "<h1>No Outputs Yet</h1>")]
        return render_template("editor/editor.html",
                               spreadsheet_data=spreadsheet_data,
                               samples=samples_data,
                               outputs_data=project_outputs)


    @app.route('/json/save/spreadsheet', methods=['POST'])
    @login_required
    def json_save_spreadsheet():
        filename = "spreadsheet.xlsx"
        filepath = os.path.join("temp", filename)
        session["last_uploaded_file"] = filename

        data = request.get_json()['jsonData']
        wb = openpyxl.Workbook()
        ws = wb.active
        for col_idx, column in enumerate(data, start=1):
            for row_idx, value in enumerate(column, start=1):
                ws.cell(row=row_idx, column=col_idx, value=value)
        wb.save(filepath)
        if session.get("open_project", 0) is not 0:
            json_data = request.get_json()['jsonData']
            project_id = session["open_project"]
            data = json_data.get("data", 0)
            file = database.get_file(project_id)
            project_content = file.content
            try:
                project_json = json.loads(project_content)
                project_json["data"] = spreadsheet.array_to_text(data)
                updated_project_content = json.dumps(project_json)
                database.write_file(project_id, updated_project_content)
                return jsonify({"success": True})
            except Exception as e:
                print(f"Error updating project data: {e}")
                return jsonify({"success": False, "error": str(e)})

        return jsonify({"result": "ok", "filename": "filename"})

    @app.route('/get_sample_names', methods=['POST'])
    @login_required
    def get_sample_names():
        filename = "spreadsheet.xlsx"
        filepath = os.path.join("temp", filename)
        session["last_uploaded_file"] = filename

        data = request.get_json()['jsonData']
        wb = openpyxl.Workbook()
        ws = wb.active
        for col_idx, column in enumerate(data, start=1):
            for row_idx, value in enumerate(column, start=1):
                ws.cell(row=row_idx, column=col_idx, value=value)
        wb.save(filepath)
        if session.get("open_project", 0) != 0:
            json_data = request.get_json()['jsonData']
            project_id = session["open_project"]
            data = json_data.get("data", 0)
            try:
                samples = spreadsheet.read_samples(data)
                sample_names = [sample.name for sample in samples]
                return jsonify({"sample_names": sample_names})
            except Exception as e:
                print(f"Error updating project data: {e}")
                return jsonify({"success": False, "error": str(e)})
        return jsonify({"sample_names": "failed"})

    @app.route('/delete_output', methods=['POST'])
    @login_required
    def del_output():
        output_name = request.form.get('output_name', '')
        output_data = request.form.get('output_data', '')
        graph_output = Output(output_name, "graph", output_data)
        matrix_output = Output(output_name, "matrix", output_data)

    @app.route('/new_output', methods=['GET'])
    @login_required
    def new_output():
        output_name = request.args.get('output_name', '')
        sample_names = request.args.getlist('samples')
        output_type = request.args.get('output_type', '')

        project_id = session.get("open_project", 0)
        file = database.get_file(project_id)
        project_content = file.content
        project_data = get_project_data(project_content)
        spreadsheet_data = spreadsheet.text_to_array(project_data)
        loaded_samples = spreadsheet.read_samples(spreadsheet_data)

        output_data = ""
        active_samples = []
        for sample in loaded_samples:
            for sample_name in sample_names:
                if sample.name == sample_name:
                    active_samples.append(sample)

        if output_type == "kde_graph":
            graph = Graph(samples=active_samples,
                          title=output_name,
                          stacked=False,
                          graph_type="kde")
            output_data = graph.generate_svg()
            output_type = "graph"
        elif output_type == "pdp_graph":
            graph = Graph(samples=active_samples,
                          title=output_name,
                          stacked=False,
                          graph_type="pdp")
            output_data = graph.generate_svg()
            output_type = "graph"
        elif output_type == "cdf_graph":
            graph = Graph(samples=active_samples,
                          title=output_name,
                          stacked=False,
                          graph_type="cdf")
            output_data = graph.generate_svg()
            output_type = "graph"
        elif output_type == "similarity_matrix":
            matrix = Matrix(active_samples, "similarity")
            output_data = matrix.to_html()
            output_type = "matrix"
        elif output_type == "likeness_matrix":
            matrix = Matrix(active_samples, "likeness")
            output_data = matrix.to_html()
            output_type = "matrix"
        elif output_type == "ks_matrix":
            matrix = Matrix(active_samples, "ks")
            output_data = matrix.to_html()
            output_type = "matrix"
        elif output_type == "kuiper_matrix":
            matrix = Matrix(active_samples, "kuiper")
            output_data = matrix.to_html()
            output_type = "matrix"
        elif output_type == "r2_matrix":
            matrix = Matrix(active_samples, "r2")
            output_data = matrix.to_html()
            output_type = "matrix"
        elif output_type == "dis_similarity_matrix":
            matrix = Matrix(active_samples, "dissimilarity")
            output_data = matrix.to_html()
            output_type = "matrix"
        elif output_type == "dis_likeness_matrix":
            matrix = Matrix(active_samples, "similarity")
            output_data = matrix.to_html()
            output_type = "matrix"
        elif output_type == "dis_ks_matrix":
            matrix = Matrix(active_samples, "similarity")
            output_data = matrix.to_html()
            output_type = "matrix"
        elif output_type == "dis_kuiper_matrix":
            matrix = Matrix(active_samples, "similarity")
            output_data = matrix.to_html()
            output_type = "matrix"
        elif output_type == "dis_r2_matrix":
            matrix = Matrix(active_samples, "similarity")
            output_data = matrix.to_html()
            output_type = "matrix"

        if get_all_outputs(project_content) is None:
            outputs = []
        else:
            outputs = get_all_outputs(project_content)

        output = Output(output_name, output_type, output_data)
        outputs.append(output)

        updated_project_content = set_all_outputs(project_content, outputs)
        database.write_file(project_id, updated_project_content)
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
            output_name = output.get("output_name")
            output_type = output.get("output_type")
            output_data = output.get("output_data")
            outputs.append(Output(output_name, output_type, output_data))
        return outputs
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None


def set_all_outputs(json_string, outputs):
    name = get_project_name(json_string)
    data = get_project_data(json_string)
    updated_project = Project(name=name, data=data, outputs=outputs)
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


def get_project_name(json_string):
    try:
        data = json.loads(json_string)
        project_name = data.get("project_name")
        return project_name
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None
