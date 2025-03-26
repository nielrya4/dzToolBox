import os
from flask import request, jsonify
from utils import spreadsheet
from dz_lib.utils import data
from werkzeug.utils import secure_filename
import json
from api.account import token_required
import numpy as np

def register(app):
    @app.route('/api/data/samples-from-xlsx', methods=['POST'])
    @token_required
    def samples(current_user):
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if not file.filename.endswith('.xlsx'):
            return jsonify({"error": "Invalid file type. Please upload an Excel file."}), 400

        filename = secure_filename(file.filename)
        temp_file_path = os.path.join('/tmp', filename)
        file.save(temp_file_path)

        try:
            spreadsheet_data = spreadsheet.excel_to_array(temp_file_path)
            spreadsheet_data = np.transpose(spreadsheet_data)
        except Exception as e:
            return jsonify({"error": f"Error reading the Excel file: {str(e)}"}), 500
        loaded_samples = data.read_1d_samples(spreadsheet_data)
        loaded_samples_dict = [sample.to_dict() for sample in loaded_samples]
        loaded_samples_json = json.dumps(loaded_samples_dict, indent=4)
        return jsonify({"samples": loaded_samples_json})

    @app.route('/api/data/subset-of-samples', methods=['POST'])
    @token_required
    def subset_of_samples(current_user):
        try:
            request_data = request.get_json()
            if not request_data:
                return jsonify({"error": "Invalid JSON data."}), 400

            samples_json = request_data.get("samples")
            sample_names = request_data.get("sample_names")

            if not samples_json or not sample_names:
                return jsonify({"error": "Missing 'samples' or 'sample_names' in request data."}), 400

            samples = json.loads(samples_json)
            subset = [sample for sample in samples if sample.get("name") in sample_names]

            return jsonify({"subset": json.dumps(subset, indent=4)})

        except Exception as e:
            return jsonify({"error": f"Error processing subset: {str(e)}"}), 500
