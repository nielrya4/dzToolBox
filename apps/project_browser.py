from flask import render_template, request, session, jsonify
from flask_login import login_required
from server import database, files
from jinja2_fragments import render_block
from jinja2 import Environment, FileSystemLoader, select_autoescape
from utils.project import Project
from utils import spreadsheet
import numpy as np
import json

environment = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(("html", "jinja2"))
)


def register(app):

    @app.route('/projects', methods=['GET', 'POST'])
    @login_required
    def projects():
        user_projects = database.get_all_files()
        return render_template('project_browser/project_browser.html', user_projects=user_projects)

    @app.route('/delete_project/<int:project_id>', methods=['POST'])
    @login_required
    def delete_project(project_id):
        database.delete_file(project_id)
        return render_project_list()

    @app.route('/new_project', methods=['GET', 'POST'])
    @login_required
    def new_project():
        project_name = request.form.get('project_name', "New Project")
        file = request.files['data_file']
        if file is not None:
            spreadsheet_array = spreadsheet.excel_to_array(file)
            spreadsheet_transposed = np.transpose(spreadsheet_array)
            spreadsheet_list = np.ndarray.tolist(spreadsheet_transposed)
            spreadsheet_data = json.dumps(spreadsheet_list)
        else:
            spreadsheet_data = "<h1>No Data</h1>"
        project_data = Project(name=project_name,
                               data=spreadsheet_data,
                               outputs="").generate_json_string()
        file = database.new_file(project_name, project_data)
        session["open_project"] = 0
        return render_project_list()


def render_project_list():
    user_projects = database.get_all_files()
    project_list_html = render_block(environment=environment,
                                     template_name="project_browser/project_browser.html",
                                     block_name="project_list",
                                     user_projects=user_projects)
    return project_list_html

