"""
Multivariate spreadsheet routes - save and sample names
"""

import base64
import zlib
import json
from flask import request, jsonify, session
from flask_login import login_required
from server import database
from utils import spreadsheet, compression
from utils.project import project_from_json


def __get_project(project_id):
    if session.get("open_project", 0) == project_id:
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        return project_from_json(project_content)
    else:
        return None


def register(app):

    @app.route('/projects/<int:project_id>/multivariate/save', methods=['POST'])
    @login_required
    def save_grainalyzer_data(project_id):
        """Save multivariate spreadsheet data"""
        if session.get("open_project", 0) == project_id:
            try:
                compressed_data = request.get_json().get('compressedData', '')
                if not compressed_data:
                    return jsonify({"success": False, "error": "No compressed data provided"})

                compressed_data_bytes = base64.b64decode(compressed_data)
                decompressed_data = zlib.decompress(compressed_data_bytes).decode('utf-8')
                json_data = json.loads(decompressed_data)
                data = json_data.get("data", [])

                if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
                    raise ValueError("Data is not in the expected list of lists format.")

                project = __get_project(project_id)
                project.grainalyzer_data = spreadsheet.array_to_text(data)

                compressed_proj_content = compression.compress(project.to_json())
                database.write_file(project_id, compressed_proj_content)

                return jsonify({"success": True})
            except Exception as e:
                print(f"Error saving multivariate data: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({"success": False, "error": str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/tensor-sample-names', methods=['GET'])
    @login_required
    def get_tensor_sample_names(project_id):
        """Get sample names for multivariate tensor factorization"""
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            spreadsheet_data = spreadsheet.text_to_array(project.grainalyzer_data)
            try:
                samples, feature_names = spreadsheet.read_multivariate_samples(
                    spreadsheet_array=spreadsheet_data,
                    max_age=4500
                )
                sample_names = [sample.name for sample in samples]
                return jsonify({"sample_names": sample_names, "feature_names": feature_names})
            except Exception as e:
                print(f"Error reading multivariate sample names: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({"success": False, "error": str(e)})
        else:
            return jsonify({"error": "access_denied"}), 403
