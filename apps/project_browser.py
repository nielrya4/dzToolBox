from flask import render_template, request, session, jsonify
from flask_login import login_required
from server import database, files
from jinja2_fragments import render_block
from jinja2 import Environment, FileSystemLoader, select_autoescape
from utils.project import Project
from utils.spreadsheet import extract_data_from_file
import os
import app as APP
import openpyxl

environment = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(("html", "jinja2"))
)


def register(app):

    @app.route('/project_browser', methods=['GET', 'POST'])
    @login_required
    def project_browser():
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
        if request.method == 'POST':
            project_name = request.form.get('project_name')
            project_name = "New Project" if project_name == '' else project_name
            file = session.get('last_uploaded_file', None)
            if file is not None:
                spreadsheet_data = str(extract_data_from_file(os.path.join('temp', file)))
            else:
                spreadsheet_data = "<h1>No Data</h1>"

            project_data = Project(name=project_name,
                                   data=spreadsheet_data,
                                   outputs="").generate_xml_data()
            database.new_file(project_name, project_data)
        return render_project_list()

    @app.route('/json/save', methods=['POST'])
    def json_save():
        data = request.get_json()['jsonData']

        wb = openpyxl.Workbook()
        ws = wb.active

        # Write the data to the worksheet
        for col_idx, column in enumerate(data, start=1):
            for row_idx, value in enumerate(column, start=1):
                ws.cell(row=row_idx, column=col_idx, value=value)

        # Save the workbook to a file in the temp folder
        if session.get('last_uploaded_file') is None:
            filename = 'spreadsheet.xlsx'
        else:
            filename = session['last_uploaded_file']
        filepath = os.path.join('temp', filename)
        session['last_uploaded_file'] = filename

        wb.save(filepath)
        return jsonify({"result": "ok", "filename": filename})


def render_project_list():
    user_projects = database.get_all_files()
    project_list_html = render_block(environment=environment,
                                     template_name="project_browser/project_browser.html",
                                     block_name="project_list",
                                     user_projects=user_projects)
    return project_list_html

