from flask import render_template, send_from_directory
from flask_login import login_required
import app as APP
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

environment = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(("html", "jinja2"))
)


def register(app):
    db = APP.db

    @app.route('/docs')
    def docs():
        return render_template("docs/docs.html")

    @app.route('/docs/api')
    def api_docs():
        return render_template("docs/api_docs.html")

    @app.route('/docs/gsa_poster.pdf', methods=['GET', 'POST'])
    def gsa_poster():
        return send_from_directory(os.path.join(app.root_path, 'static'), 'docs/poster_final.pdf',
                                   mimetype='application/pdf')
