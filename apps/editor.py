from flask import render_template, request, jsonify
from flask_login import login_required, current_user
from server import database
from jinja2 import Environment, FileSystemLoader, select_autoescape
from utils import _xml, spreadsheet
import xml.etree.ElementTree as ET


environment = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(("html", "jinja2"))
)


def register(app):
    @app.route('/open_project/<int:project_id>', methods=['GET', 'POST'])
    @login_required
    def open_project(project_id):
        file = database.get_file(project_id)
        project_content = file.content

        spreadsheet_content = _xml.extract_data(project_content, "data")
        outputs_content = _xml.extract_data(project_content, "outputs")

        # spreadsheet_data = spreadsheet.text_to_array(spreadsheet_content)
        spreadsheet_data = spreadsheet_content # TODO delete this line but then make the spreadsheet data show up on the handsontable
        # outputs_data = get_all_outputs(outputs_content[:][1])
        outputs_data = "<p>asdf</p>"
        return render_template("editor/editor.html",
                               spreadsheet_data=spreadsheet_data,
                               outputs_data=outputs_data)


def get_all_outputs(xml_string):
    try:
        outputs = []
        root = ET.fromstring(xml_string)
        for output in root.findall("output"):
            output_name = output.get("name")
            output_data = output.text
            outputs.append((output_name, output_data))
        return outputs
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None


def get_output_by_name(xml_string, output_name):
    try:
        root = ET.fromstring(xml_string)
        for output in root.findall(".//output"):
            if output.get("name") == output_name:
                return output.text
        return None
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None
