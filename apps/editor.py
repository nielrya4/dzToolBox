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
    @app.route('/open_project/<int:project_id>', methods=['GET', 'POST'])
    @login_required
    def open_project(project_id):
        file = database.get_file(project_id)
        # Replace this return with opening the project
        return file.content
