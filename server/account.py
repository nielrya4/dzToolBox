from flask import render_template, request, redirect, url_for, flash
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
            return redirect(url_for('editor'))

        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')

            user = APP.User.query.filter_by(username=username).first()

            if user and check_password_hash(user.password, password):
                login_user(user)
                flash('Login successful!', 'success')
                return redirect(url_for('editor'))

            else:
                flash('Login unsuccessful. Please check your username and password.', 'error')

        return render_template('init/login.html', current_user=current_user)
