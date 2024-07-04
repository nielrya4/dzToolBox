from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin
from server import route, cleanup
import os
import secrets


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://koyeb-adm:J0OqWG2Lalmy@ep-lucky-boat-a44k24rd.us-east-1.pg.koyeb.app/koyebdb'
SECRET_KEY = secrets.token_hex(16)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['TEMP_FOLDER'] = 'temp'
if not os.path.exists('temp'):
    os.makedirs('temp')

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()

route.register_routes(app)
cleanup.start_cleanup()


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    files = db.relationship('CodeFile', backref='author', lazy=True)


class CodeFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='SET NULL'))


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
