"""
Univariate spreadsheet routes - save, export, sample names
"""

import dz_lib.utils.encode
from flask import request, jsonify, session
from flask_login import login_required
from flask import send_file
from pathvalidate import sanitize_filename
from server import database
from utils import spreadsheet, compression
from utils.project import project_from_json
from dz_lib.utils import matrices
import base64
import zlib
import json
import pandas as pd


def __get_project(project_id):
    if session.get("open_project", 0) == project_id:
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        return project_from_json(project_content)
    else:
        return None


def __is_float(element) -> bool:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def register(app):

    @app.route('/projects/<int:project_id>/save', methods=['POST'])
    @login_required
    def save_project(project_id):
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
                for i, row in enumerate(data):
                    for j, cell in enumerate(row):
                        if cell is not None:
                            if str(cell).strip() == '':
                                data[i][j] = None
                            elif __is_float(cell):
                                data[i][j] = float(str(cell).strip())
                project = __get_project(project_id)
                project.data = spreadsheet.array_to_text(data)
                compressed_proj_content = compression.compress(project.to_json())
                database.write_file(project_id, compressed_proj_content)
                return jsonify({"success": True})
            except Exception as e:
                print(f"Error updating project data: {e}")
                return jsonify({"success": False, "error": str(e)})
        else:
            return jsonify({"save": "access_denied"})

    @app.route('/projects/<int:project_id>/sample-names', methods=['GET'])
    @login_required
    def get_sample_names(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            spreadsheet_data = spreadsheet.text_to_array(project.data)
            try:
                samples = spreadsheet.read_samples(spreadsheet_data)
                sample_names = [sample.name for sample in samples]
                return jsonify({"sample_names": sample_names})
            except Exception as e:
                print(f"Error reading sample names: {e}")
                return jsonify({"success": False, "error": str(e)})
        else:
            return jsonify({"sample-names": "access_denied"})

    @app.route('/projects/<int:project_id>/data/export', methods=['GET'])
    @login_required
    def export_data(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            spreadsheet_data = spreadsheet.text_to_array(project.data)
            filename = sanitize_filename(request.args.get("filename", "exported_data"))
            file_format = request.args.get("format", "xlsx")
            try:
                df = pd.DataFrame(spreadsheet_data)
                if file_format == "xlsx":
                    buffer = matrices.to_xlsx(df, include_header=False, include_index=False)
                    return send_file(
                        buffer,
                        as_attachment=True,
                        download_name=f"{filename}.xlsx",
                        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif file_format == "csv":
                    buffer = matrices.to_csv(df, include_header=False, include_index=False)
                    buffer.seek(0)
                    return send_file(
                        buffer,
                        as_attachment=True,
                        download_name=f"{filename}.csv",
                        mimetype="text/csv"
                    )
                elif file_format == "xls":
                    buffer = matrices.to_xls(df, include_header=False, include_index=False)
                    buffer.seek(0)
                    return send_file(
                        buffer,
                        as_attachment=True,
                        download_name=f"{filename}.xls",
                        mimetype=dz_lib.utils.encode.get_mime_type("xls")
                    )
                elif file_format == "json":
                    json_data = df.to_json(orient='values')
                    return jsonify(json_data)
                return jsonify({"success": False, "error": "Unsupported format"})
            except Exception as e:
                print(f"Error exporting data: {e}")
                return jsonify({"success": False, "error": str(e)})
        else:
            return jsonify({"error": "access_denied"})
