from flask import render_template


def register(app):
    @app.route('/')
    def home():
        return render_template('init/home.html')