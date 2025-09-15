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
    'pool_size': 20,  # Base connection pool size
    'max_overflow': 30,  # Additional connections during peak load
    'pool_recycle': 3600,  # Recycle connections every hour
    'pool_timeout': 30,  # Timeout when getting connection from pool
}

SECRET_KEY = secrets.token_hex(16)
app.config['SECRET_KEY'] = SECRET_KEY

app.config['TEMP_FOLDER'] = 'temp'
if not os.path.exists('temp'):
    os.makedirs('temp')

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(20), unique=True, nullable=False, index=True)
    password = db.Column(db.String(60), nullable=False)
    files = db.relationship('CodeFile', backref='author', lazy=True)


class CodeFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False, index=True)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), index=True)
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