from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin
from flask_migrate import Migrate
from sqlalchemy.exc import SQLAlchemyError
from server import route, cleanup
import os
import secrets
from dotenv import load_dotenv
from flask_cors import CORS



app = Flask(__name__)
CORS(app)
load_dotenv()
database_url = os.getenv('DATABASE_URL', 'not found')
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f'{database_url}'
    '?sslmode=require'
)

app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    # uwsgi runs 4 processes x 2 threads, so 8 requests at once at most; the database allows
    # 225 connections in total. 20+30 per process (~200 across 4 processes) could exhaust it.
    'pool_size': 5,  # Base connection pool size per process
    'max_overflow': 5,  # Additional connections during peaks
    'pool_recycle': 3600,  # Recycle connections every hour
    'pool_timeout': 30,  # Timeout when getting connection from pool
}

# Set SECRET_KEY in .env so sessions/API tokens are stable across restarts and workers.
# (Generate one with: python -c "import secrets; print(secrets.token_hex(32))")
SECRET_KEY = os.getenv('SECRET_KEY')
if not SECRET_KEY:
    SECRET_KEY = secrets.token_hex(32)
    print("WARNING: SECRET_KEY is not set; using a random key. All sessions will be invalidated on every restart.")
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', '0') == '1'  # set to 1 behind HTTPS

app.config['TEMP_FOLDER'] = 'temp'
if not os.path.exists('temp'):
    os.makedirs('temp')

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    # Guest accounts are temporary and deleted nightly. Never infer this from the username.
    is_guest = db.Column(db.Boolean, nullable=False, default=False, server_default=db.false())
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    # passive_deletes: the database's ON DELETE CASCADE removes a user's files
    files = db.relationship('CodeFile', backref='author', lazy=True, passive_deletes=True)


# Usernames are also unique case-insensitively ("Nancy@x.edu" and "nancy@x.edu" are the same person).
db.Index('user_username_lower_key', db.func.lower(User.username), unique=True)


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None


class CodeFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False, index=True)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp(), index=True)
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())
    
    __table_args__ = (
        db.Index('idx_user_title', 'user_id', 'title'),
        db.Index('idx_user_created', 'user_id', 'created_at'),
        db.Index('idx_user_updated', 'user_id', 'updated_at'),
    )


with app.app_context():
    try:
        db.create_all()
    except SQLAlchemyError as e:
        db.session.rollback()
        print(f"Error during db.create_all: {e}")
    finally:
        db.session.close()

route.register_routes(app)
cleanup.start_cleanup()


if __name__ == '__main__':
    try:
        app.run(host="0.0.0.0")
    except Exception as e:
        print(f"Error starting the Flask application: {e}")