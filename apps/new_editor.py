from flask import render_template, request, jsonify, session
from flask_login import login_required, current_user
from server import database
from utils import spreadsheet, compression
from utils.output import Output
from utils.project import project_from_json
import json


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

            grainalyzer_data_parsed = "[[null]]"
            if project.grainalyzer_data and project.grainalyzer_data.strip():
                grainalyzer_data_parsed = project.grainalyzer_data

            return render_template("editor/editor.html",
                                   spreadsheet_data=project.data,
                                   samples=samples_data,
                                   outputs_data=project.outputs,
                                   multivariate_spreadsheet_data=grainalyzer_data_parsed,
                                   multivariate_outputs_data=project.grainalyzer_outputs,
                                   project_id=project_id,
                                   project_name=project.name)
        else:
            return render_template("errors/403.html")

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


def __get_project(project_id):
    if session.get("open_project", 0) == project_id:
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        return project_from_json(project_content)
    else:
        return None
