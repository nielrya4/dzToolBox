import numpy as np
from flask import render_template, request, redirect, url_for, flash, session
from flask_login import UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import app as APP
from jinja2_fragments import render_block
from jinja2 import Environment, FileSystemLoader, select_autoescape
import secrets
from utils.project import Project
from server import database
from utils import spreadsheet

environment = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(("html", "jinja2"))
)


def register(app):
    db = APP.db
    login_manager = APP.login_manager

    @login_manager.user_loader
    def load_user(user_id):
        return APP.User.query.get(int(user_id))

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('home'))

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')

            if password != confirm_password:
                flash('Passwords do not match', 'error')
                return redirect(url_for('signup'))
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

            new_user = APP.User(username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()

            flash('Account created successfully. You can now log in.', 'success')
            return redirect(url_for('login'))

        return render_template('init/signup.html')

    # Create Guest Account: Creates a temporary account, logs in, creates the first project, and redirects to the editor
    @app.route('/create_guest_account')
    def create_guest_account():
        if current_user.is_authenticated:
            logout_user()
        secret_key = secrets.token_hex(16)
        username = str(secret_key) + "_guest"
        password = "guest"
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        guest_user = APP.User(username=username, password=hashed_password)
        db.session.add(guest_user)
        db.session.commit()
        user = APP.User.query.filter_by(username=username).first()
        login_user(user)
        session["user_id"] = user.id
        project_name = "Default Project"
        spreadsheet_data = spreadsheet.array_to_text([[None] * 6] * 6)
        project_data = Project(name=project_name,
                               data=spreadsheet_data,
                               outputs="").generate_json_string()
        file = database.new_file(project_name, project_data)
        session["open_project"] = file.id
        return redirect(f"/projects/{file.id}")

    @app.route('/delete_account', methods=['GET', 'POST'])
    @login_required
    def delete_account():
        if request.method == 'POST':
            # Delete associated code files
            APP.CodeFile.query.filter_by(user_id=current_user.id).delete()

            # Delete the user
            db.session.delete(current_user)
            db.session.commit()

            # Log out the user after deleting the account
            logout_user()

            flash('Your account has been deleted.', 'success')
            return redirect(url_for('login'))  # Redirect to the login page

        return render_template('init/delete_account.html', current_user=current_user)

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('projects'))

        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')

            user = APP.User.query.filter_by(username=username).first()

            if user and check_password_hash(user.password, password):
                login_user(user)
                flash('Login successful!', 'success')
                session["user_id"] = user.id
                return redirect(url_for('projects'))

            else:
                flash('Login unsuccessful. Please check your username and password.', 'error')

        return render_template('init/login.html', current_user=current_user)

    @app.route('/')
    def home():
        if current_user.is_authenticated:
            return redirect(url_for('projects'))
        return render_template('init/home.html')
