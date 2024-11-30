from json import loads, dumps
from flask import render_template
from flask_login import login_required
from server import database
from utils.project import Settings
from utils import compression


def register(app):
    @app.route('/update-projects', methods=['GET', 'POST'])
    @login_required
    def update_projects():
        user_projects = database.get_all_files()
        sorted_user_projects = sorted(user_projects, key=lambda x: str.lower(x.title))
        for file in sorted_user_projects:
            project_content = compression.decompress(file.content)
            json_data = loads(project_content)
            json_data["settings"] = Settings().to_json()
            updated_content = dumps(json_data, indent=4)
            compressed_proj_content = compression.compress(updated_content)
            database.write_file(file.id, compressed_proj_content)
        return render_template('project_browser/project_browser.html', user_projects=sorted_user_projects)
