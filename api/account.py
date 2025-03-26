import jwt
import datetime
from flask import Flask, request, jsonify, session
from flask_login import LoginManager, login_user, current_user, logout_user
from werkzeug.security import check_password_hash
import dzToolBox as APP
import jwt
from functools import wraps

def register(app):
    login_manager = APP.login_manager

    @login_manager.user_loader
    def api_load_user(user_id):
        return APP.User.query.get(int(user_id))

    # Login endpoint with JWT
    @app.route('/api/login', methods=['POST'])
    def api_login():
        if current_user.is_authenticated:
            return jsonify({'message': 'Already logged in.'}), 200

        username = request.json.get('username')
        password = request.json.get('password')

        if not username or not password:
            return jsonify({'error': 'Missing username or password.'}), 400

        try:
            user = APP.User.query.filter_by(username=username).first()
        except Exception as e:
            return jsonify({'error': f"Error querying user: {str(e)}"}), 500

        if user and check_password_hash(user.password, password):
            login_user(user)

            # Generate JWT token
            token = jwt.encode({
                'user_id': user.id,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
            }, APP.SECRET_KEY, algorithm='HS256')

            return jsonify({'token': token}), 200
        else:
            return jsonify({'error': 'Invalid username or password.'}), 401

    # Logout endpoint
    @app.route('/api/logout', methods=['POST'])
    def api_logout():
        logout_user()
        return jsonify({'message': 'Logged out successfully.'}), 200


def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        if not token:
            return jsonify({'error': 'Token is missing!'}), 403
        try:
            data = jwt.decode(token, APP.SECRET_KEY, algorithms=["HS256"])
            current_user = data['user_id']
        except Exception as e:
            return jsonify({'error': 'Token is invalid or expired!'}), 403
        return f(current_user, *args, **kwargs)
    return decorated_function