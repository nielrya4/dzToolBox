from flask import render_template, request, redirect, url_for, flash, session
from flask_login import UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import app as APP
from jinja2_fragments import render_block
from jinja2 import Environment, FileSystemLoader, select_autoescape

environment = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(("html", "jinja2"))
)


def register(app):
    db = APP.db

    @app.route('/docs')
    @login_required
    def docs():
        return render_template("docs/docs.html")
