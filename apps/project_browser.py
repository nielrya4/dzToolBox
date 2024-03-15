from flask import render_template, request, jsonify
from flask_login import login_required, current_user
import app as APP
from server import database
from jinja2_fragments import render_block
from jinja2 import Environment, FileSystemLoader, select_autoescape

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

    @app.route('/new_project', methods=['POST'])
    @login_required
    def new_project():
        if request.method == 'POST':
            project_name = request.form.get('project_name')
            project_name = "New Project" if project_name == '' else project_name
            data_file = request.form.get('data_file', '')
            data = '<h1>hello world</h1>'
            database.new_file(project_name, data)
        return render_project_list()


def render_project_list():
    user_projects = database.get_all_files()
    project_list_html = render_block(environment=environment,
                                  template_name="project_browser/project_browser.html",
                                  block_name="project_list",
                                  user_projects=user_projects)
    return project_list_html
