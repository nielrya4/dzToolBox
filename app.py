from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin
from sqlalchemy.exc import SQLAlchemyError
from server import route, cleanup
import os
import secrets


app = Flask(__name__)

# Improved connection string with SSL mode explicitly set
app.config['SQLALCHEMY_DATABASE_URI'] = (
    'postgresql://koyeb-adm:J0OqWG2Lalmy@ep-lucky-boat-a44k24rd.us-east-1.pg.koyeb.app/koyebdb'
    '?sslmode=require'
)

# Enable pool_pre_ping to handle stale connections
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True}

# Secret key management
SECRET_KEY = secrets.token_hex(16)
app.config['SECRET_KEY'] = SECRET_KEY

# Temporary folder setup
app.config['TEMP_FOLDER'] = 'temp'
if not os.path.exists('temp'):
    os.makedirs('temp')

# Initialize database and login manager
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


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


with app.app_context():
    # Use try-except for handling potential errors during database setup
    try:
        db.create_all()
    except SQLAlchemyError as e:
        db.session.rollback()
        print(f"Error during db.create_all: {e}")
    finally:
        db.session.close()

# Register routes and start cleanup processes
route.register_routes(app)
cleanup.start_cleanup()


if __name__ == '__main__':
    # Additional try-except to catch errors during app initialization
    try:
        app.run(host="0.0.0.0")
    except Exception as e:
        print(f"Error starting the Flask application: {e}")
