import os.path
from flask import render_template, request, redirect, url_for, flash, session
from flask_login import login_user, login_required, logout_user, current_user
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash
import dzToolBox as APP
from jinja2 import Environment, FileSystemLoader, select_autoescape
import secrets
from utils.project import Project
from server import database
from utils import spreadsheet, compression

environment = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(("html", "jinja2"))
)


def register(app):
    db = APP.db

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('home'))

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if request.method == 'POST':
            username = (request.form.get('username') or '').strip()
            password = request.form.get('password') or ''
            confirm_password = request.form.get('confirm_password') or ''

            if not username or not password:
                flash('Username and password are required', 'error')
                return redirect(url_for('signup'))
            if len(username) > 254:
                flash('Username is too long', 'error')
                return redirect(url_for('signup'))
            if username.lower().endswith('_guest'):
                flash('That username is reserved. Please choose another.', 'error')
                return redirect(url_for('signup'))
            if password != confirm_password:
                flash('Passwords do not match', 'error')
                return redirect(url_for('signup'))
            if database.get_user_by_username(username) is not None:
                flash('That username is already taken. If it is yours, please log in instead.', 'error')
                return redirect(url_for('signup'))

            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = APP.User(username=username, password=hashed_password)
            db.session.add(new_user)
            try:
                db.session.commit()
            except IntegrityError:
                # Lost a race with another signup for the same name; the database constraint caught it.
                db.session.rollback()
                flash('That username is already taken. If it is yours, please log in instead.', 'error')
                return redirect(url_for('signup'))

            flash('Account created successfully. You can now log in.', 'success')
            return redirect(url_for('login'))

        return render_template('init/pages/signup.html')

    # Create Guest Account: Creates a temporary account, logs in, creates the first project, and redirects to the editor
    @app.route('/create_guest_account')
    def create_guest_account():
        if current_user.is_authenticated:
            logout_user()
        secret_key = secrets.token_hex(16)
        username = str(secret_key) + "_guest"
        # Nobody ever logs into a guest account by password, so make it unguessable.
        hashed_password = generate_password_hash(secrets.token_urlsafe(32), method='pbkdf2:sha256')
        guest_user = APP.User(username=username, password=hashed_password, is_guest=True)
        db.session.add(guest_user)
        db.session.commit()
        user = guest_user
        login_user(user)
        session["user_id"] = user.id
        project_name = "Default Project"
        spreadsheet_file = os.path.join('static', 'global', 'docs', 'complex_synthetic.xlsx')
        spreadsheet_array = spreadsheet.excel_to_array(spreadsheet_file)
        transposed_array = [[spreadsheet_array[j][i] for j in range(len(spreadsheet_array))] for i in range(len(spreadsheet_array[0]))]
        spreadsheet_data = spreadsheet.array_to_text(transposed_array)
        project_data = Project(name=project_name,
                               data=spreadsheet_data,
                               outputs="").to_json()
        compressed_project = compression.compress(project_data)
        file = database.new_file(project_name, compressed_project)
        session["open_project"] = file.id
        return redirect(f"/projects/{file.id}")

    @app.route('/delete_account', methods=['GET', 'POST'])
    @login_required
    def delete_account():
        if request.method == 'POST':
            APP.CodeFile.query.filter_by(user_id=current_user.id).delete()

            db.session.delete(current_user)
            db.session.commit()
            logout_user()

            flash('Your account has been deleted.', 'success')
            return redirect(url_for('login'))  # Redirect to the login page

        return render_template('init/pages/delete_account.html', current_user=current_user)

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('projects'))
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password') or ''
            try:
                user = database.get_user_by_username(username)
            except Exception as e:
                login_message = f"Error querying user by username. This means the connection with the database was a little whack. Check your internet connection. Just try again, as it usually works on the second or third attempt."
                return render_template('init/pages/login.html', login_message=login_message, current_user=current_user)

            if user and check_password_hash(user.password, password):
                login_user(user)
                flash('Login successful!', 'success')
                session["user_id"] = user.id
                return redirect(url_for('projects'))
            else:
                login_message = f'Login unsuccessful. Please check your username and password.'
                return render_template('init/pages/login.html', login_message=login_message, current_user=current_user)
        else:
            return render_template('init/pages/login.html', current_user=current_user)

    @app.route('/')
    def home():
        if current_user.is_authenticated:
            return redirect(url_for('projects'))
        return render_template('init/pages/home.html')
