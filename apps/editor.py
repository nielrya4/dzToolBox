from flask import render_template, request, jsonify, session
from flask_login import login_required, current_user
from server import database
from jinja2 import Environment, FileSystemLoader, select_autoescape
from utils import spreadsheet
import json
import openpyxl
import os


environment = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(("html", "jinja2"))
)


def register(app):
    @app.route('/open_project/<int:project_id>', methods=['GET', 'POST'])
    @login_required
    def open_project(project_id):
        session["open_project"] = project_id
        file = database.get_file(project_id)
        project_content = file.content
        project_data = get_project_data(project_content)
        spreadsheet_data = spreadsheet.text_to_array(project_data)
        # outputs_data = get_all_outputs(project_content)[:][1]
        outputs_data = "asdf"
        return render_template("editor/editor.html",
                               spreadsheet_data=spreadsheet_data,
                               outputs_data=outputs_data)

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


def get_all_outputs(json_string):
    try:
        outputs = []
        data = json.loads(json_string)
        for output in data.get("outputs", []):
            output_name = output.get("output_name")
            output_data = output.get("output_data")
            outputs.append((output_name, output_data))
        return outputs
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None


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
