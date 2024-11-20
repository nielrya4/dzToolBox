from flask import render_template, request, session, jsonify
from flask_login import login_required
from server import database, files
from jinja2_fragments import render_block
from jinja2 import Environment, FileSystemLoader, select_autoescape
from utils.project import Project
from utils import spreadsheet, compression
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
        sorted_user_projects = sorted(user_projects, key=lambda x: str.lower(x.title))
        return render_template('project_browser/project_browser.html', user_projects=sorted_user_projects)

    @app.route('/delete_project/<int:project_id>', methods=['POST'])
    @login_required
    def delete_project(project_id):
        database.delete_file(project_id)
        return render_project_list()

    @app.route('/new_project', methods=['GET', 'POST'])
    @login_required
    def new_project():
        project_name = request.form.get('project_name', "New Project")
        file = request.files.get('data_file', None)
        if file is not None:
            spreadsheet_array = spreadsheet.excel_to_array(file)
            spreadsheet_transposed = np.transpose(spreadsheet_array)
            spreadsheet_list = np.ndarray.tolist(spreadsheet_transposed)
            spreadsheet_data = json.dumps(spreadsheet_list)
        else:
            spreadsheet_data = json.dumps(np.ndarray.tolist(np.array([[None]*6]*6)))
        project_data = Project(name=project_name,
                               data=spreadsheet_data,
                               outputs="").to_json()
        compressed_project = compression.compress(project_data)
        file = database.new_file(project_name, compressed_project)
        session["open_project"] = 0
        return render_project_list()


def render_project_list():
    user_projects = database.get_all_files()
    sorted_user_projects = sorted(user_projects, key=lambda x: str.lower(x.title))
    project_list_html = render_block(environment=environment,
                                     template_name="project_browser/project_browser.html",
                                     block_name="project_list",
                                     user_projects=sorted_user_projects)
    return project_list_html

