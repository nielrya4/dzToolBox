import pandas as pd
from flask import jsonify, current_app
from flask_login import login_required
from functools import lru_cache
import os


# Cache database files since they don't change often
@lru_cache(maxsize=10)
def _load_database_file(file_path):
    """Load database file with caching since these files are static"""
    df_sheets = pd.read_excel(file_path, sheet_name=None, header=None, engine='openpyxl')
    first_sheet_name, first_df = next(iter(df_sheets.items()))
    first_df = first_df.dropna(how="all").fillna("").reset_index(drop=True)
    return first_df.values.tolist()


def register(app):
    @app.route('/databases/<string:file>', methods=['GET'])
    @login_required
    def load_database(file):
        # Validate file name for security
        allowed_files = ['belt-dz', 'world-ig', 'world-dz']
        if file not in allowed_files:
            return jsonify({"error": "Database not found"}), 404
            
        file_path = os.path.join(current_app.root_path, 'static', 'databases', f"{file}.xlsx")
        
        if not os.path.exists(file_path):
            return jsonify({"error": "Database file not found"}), 404
            
        try:
            data = _load_database_file(file_path)
            return jsonify(data)
        except Exception as e:
            current_app.logger.error(f"Error loading database {file}: {str(e)}")
            return jsonify({"error": "Error loading database"}), 500
