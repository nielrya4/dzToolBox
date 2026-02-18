"""
Maps module - handles map visualization routes
"""

import base64
import datetime
import os
import uuid
from flask import request, jsonify, session
from flask import send_file
from flask_login import login_required
from werkzeug.utils import secure_filename
from server import database
from utils import compression
from utils.project import project_from_json


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'}


def __get_project(project_id):
    if session.get("open_project", 0) == project_id:
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        return project_from_json(project_content)
    else:
        return None


def __allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def __generate_static_map_html(map_data):
    points = map_data.get('points', [])
    center = map_data.get('center', {'lat': 40.7128, 'lng': -74.0060})
    zoom = map_data.get('zoom', 10)
    settings = map_data.get('settings', {})

    map_title = settings.get('title', 'Interactive Map Export')
    map_subtitle = settings.get('subtitle', '')
    map_footer = settings.get('footer', 'This is a static export of your interactive map.')

    markers_js = ""
    for point in points:
        popup_content = '<div class="point-info">'
        if point.get('image_url'):
            try:
                if point['image_url'].startswith('data:image'):
                    popup_content += f'<img src="{point["image_url"]}" alt="{point["title"]}" class="popup-image" style="max-width:100%; max-height:150px; cursor:pointer;" onclick="enlargeImage(this)">'
                elif point['image_url'].startswith('/static/uploads/'):
                    file_path = point['image_url'].replace('/static/', 'static/')
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as img_file:
                            img_data = img_file.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            ext = file_path.lower().split('.')[-1]
                            mime_type_map = {
                                'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
                                'gif': 'image/gif', 'webp': 'image/webp', 'svg': 'image/svg+xml'
                            }
                            mime_type = mime_type_map.get(ext, 'image/jpeg')
                            data_uri = f'data:{mime_type};base64,{img_base64}'
                            popup_content += f'<img src="{data_uri}" alt="{point["title"]}" class="popup-image" style="max-width:100%; max-height:150px; cursor:pointer;" onclick="enlargeImage(this)">'
                else:
                    popup_content += f'<img src="{point["image_url"]}" alt="{point["title"]}" class="popup-image" style="max-width:100%; max-height:150px; cursor:pointer;" onclick="enlargeImage(this)">'
            except Exception as e:
                print(f"Error encoding image for point {point.get('title', 'Unknown')}: {e}")

        popup_content += f'<h6>{point["title"]}</h6>'
        popup_content += f'<small>Lat: {point["latitude"]}, Lng: {point["longitude"]}</small>'
        popup_content += '</div>'
        popup_content_escaped = popup_content.replace('`', '\\`').replace('${', '\\${')

        markers_js += f"""
        L.marker([{point['latitude']}, {point['longitude']}])
            .addTo(map)
            .bindPopup(`{popup_content_escaped}`);
        """

    export_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{map_title}</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css" rel="stylesheet">
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
        #map {{ height: 600px; width: 100%; }}
        .header {{ text-align: center; background: #007bff; color: white; padding: 1rem; }}
        .footer {{ text-align: center; padding: 1rem; background: #f8f9fa; }}
        .image-modal {{ display: none; position: fixed; z-index: 10000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.9); backdrop-filter: blur(5px); }}
        .image-modal.show {{ display: flex; align-items: center; justify-content: center; animation: fadeIn 0.3s ease-out; }}
        .modal-content {{ position: relative; max-width: 90vw; max-height: 90vh; margin: auto; }}
        .enlarged-image {{ max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; box-shadow: 0 8px 32px rgba(0,0,0,0.5); }}
        .close-button {{ position: absolute; top: -40px; right: 0; color: white; font-size: 35px; font-weight: bold; cursor: pointer; background: rgba(0,0,0,0.5); border: none; width: 45px; height: 45px; border-radius: 50%; display: flex; align-items: center; justify-content: center; transition: all 0.2s ease; }}
        .close-button:hover {{ background: rgba(0,0,0,0.8); transform: scale(1.1); }}
        .popup-image {{ transition: opacity 0.2s ease; }}
        .popup-image:hover {{ opacity: 0.8; }}
        @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    </style>
</head>
<body>
    <div id="imageModal" class="image-modal" onclick="closeImageModal(event)">
        <div class="modal-content">
            <button class="close-button" onclick="closeImageModal()">&times;</button>
            <img id="enlargedImage" class="enlarged-image" src="" alt="">
        </div>
    </div>
    <div class="header">
        <h1>{map_title}</h1>
        {f'<p>{map_subtitle}</p>' if map_subtitle else ''}
        <small>Exported on {export_id} &bull; {len(points)} point(s)</small>
    </div>
    <div id="map"></div>
    <div class="footer">{map_footer}</div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
    <script>
        const map = L.map('map').setView([{center['lat']}, {center['lng']}], {zoom});
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ attribution: '&copy; OpenStreetMap contributors' }}).addTo(map);
        {markers_js}
        function enlargeImage(img) {{
            const modal = document.getElementById('imageModal');
            document.getElementById('enlargedImage').src = img.src;
            document.getElementById('enlargedImage').alt = img.alt;
            modal.classList.add('show');
            event.stopPropagation();
        }}
        function closeImageModal(event) {{
            const modal = document.getElementById('imageModal');
            if (!event || event.target === modal || event.target.classList.contains('close-button')) {{
                modal.classList.remove('show');
            }}
        }}
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') closeImageModal();
        }});
        document.getElementById('enlargedImage').addEventListener('click', function(event) {{
            event.stopPropagation();
        }});
    </script>
</body>
</html>'''
    return html_template


def register(app):

    @app.route('/projects/<int:project_id>/maps/points', methods=['GET'])
    @login_required
    def get_map_points(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            map_points = getattr(project.settings, 'map_points', [])
            return jsonify(map_points)
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/maps/points', methods=['POST'])
    @login_required
    def add_map_point(project_id):
        if session.get("open_project", 0) == project_id:
            try:
                project = __get_project(project_id)
                data = request.get_json()

                for field in ['latitude', 'longitude', 'title']:
                    if field not in data or not data[field]:
                        return jsonify({'error': f'Missing required field: {field}'}), 400

                point = {
                    'id': str(uuid.uuid4()),
                    'latitude': float(data['latitude']),
                    'longitude': float(data['longitude']),
                    'title': data['title'],
                    'image_url': data.get('image_url', '')
                }

                map_points = getattr(project.settings, 'map_points', [])
                map_points.append(point)
                project.settings.mapping_settings.map_points = map_points

                compressed_proj_content = compression.compress(project.to_json())
                database.write_file(project_id, compressed_proj_content)
                return jsonify(point), 201
            except ValueError:
                return jsonify({'error': 'Invalid latitude or longitude'}), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/maps/points/<point_id>', methods=['DELETE'])
    @login_required
    def delete_map_point(project_id, point_id):
        if session.get("open_project", 0) == project_id:
            try:
                project = __get_project(project_id)
                map_points = getattr(project.settings, 'map_points', [])
                original_length = len(map_points)
                map_points = [p for p in map_points if p['id'] != point_id]

                if len(map_points) == original_length:
                    return jsonify({'error': 'Point not found'}), 404

                project.settings.mapping_settings.map_points = map_points
                compressed_proj_content = compression.compress(project.to_json())
                database.write_file(project_id, compressed_proj_content)
                return jsonify({'message': 'Point deleted successfully'}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/maps/upload', methods=['POST'])
    @login_required
    def upload_map_image(project_id):
        if session.get("open_project", 0) == project_id:
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                if file and __allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filename = f"{uuid.uuid4()}_{filename}"
                    uploads_dir = os.path.join('static', 'uploads')
                    os.makedirs(uploads_dir, exist_ok=True)
                    file.save(os.path.join(uploads_dir, filename))
                    return jsonify({'url': f"/static/uploads/{filename}"}), 200
                else:
                    return jsonify({'error': 'Invalid file type'}), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/maps/clear', methods=['POST'])
    @login_required
    def clear_map_points(project_id):
        if session.get("open_project", 0) == project_id:
            try:
                project = __get_project(project_id)
                project.settings.mapping_settings.map_points = []
                compressed_proj_content = compression.compress(project.to_json())
                database.write_file(project_id, compressed_proj_content)
                return jsonify({'message': 'All points cleared successfully'}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/maps/export', methods=['POST'])
    @login_required
    def export_map(project_id):
        if session.get("open_project", 0) == project_id:
            try:
                data = request.get_json()
                map_data = data.get('map_data')
                if not map_data:
                    return jsonify({'error': 'No map data provided'}), 400

                html_content = __generate_static_map_html(map_data)
                export_filename = f"static_map_{uuid.uuid4()}.html"
                export_path = os.path.join('temp')
                os.makedirs(export_path, exist_ok=True)
                full_path = os.path.join(export_path, export_filename)

                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                return send_file(full_path, as_attachment=True, download_name='static_map.html')
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403
