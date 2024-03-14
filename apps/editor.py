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

    @app.route('/editor', methods=['GET', 'POST'])
    @login_required
    def editor():
        user_files = database.get_all_files()
        return render_template('editor/editor.html', user_files=user_files)

    @app.route('/get_file/<int:file_id>', methods=['GET', 'POST'])
    @login_required
    def get_file(file_id):
        file = database.get_file(file_id)
        # Replace this return with opening the project
        return file.content

    @app.route('/delete_file/<int:file_id>', methods=['POST'])
    @login_required
    def delete_file(file_id):
        database.delete_file(file_id)
        return render_file_list()

    @app.route('/new_file', methods=['POST'])
    @login_required
    def new_file():
        if request.method == 'POST':
            title = request.form.get('title')
            content = request.form.get('content')
            if title and content:
                database.new_file(title, content)
                return render_file_list()
            else:
                return jsonify({"result": "failed"})
        return jsonify({"result": ""})


def render_file_list():
    user_files = database.get_all_files()
    file_list_html = render_block(environment=environment,
                                  template_name="editor/editor.html",
                                  block_name="file_list",
                                  user_files=user_files)
    return file_list_html
