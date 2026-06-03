from flask import render_template, send_from_directory
from flask_login import login_required
import dzToolBox as APP
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

environment = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(("html", "jinja2"))
)


def register(app):
    db = APP.db

    @app.route('/dzgrainalyzer')
    def dzgrainalyzer():
        return render_template("dzgrainalyzer/dzgrainalyzer.html")
