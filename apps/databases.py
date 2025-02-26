import flask
import pandas as pd
from flask import jsonify
from flask_login import login_required


def register(app):
    @app.route('/databases/belt_db.json', methods=['GET'])
    @login_required
    def belt_db():
        # Load all sheets from the Excel file
        df_sheets = pd.read_excel("static/databases/belt_db.xlsx", sheet_name=None, header=None)

        # Process each sheet
        sheets = []
        for index, (sheet_name, df) in enumerate(df_sheets.items()):
            df = df.dropna(how="all")  # Remove empty rows
            if df.empty:
                continue  # Skip empty sheets

            df.reset_index(drop=True, inplace=True)  # Reset index for clean JSON output

            # Convert entire DataFrame to a raw list of lists (including headers)
            sheet_data = df.fillna("").values.tolist()

            sheet_obj = {
                "title": sheet_name,  # Use sheet name as title
                "key": f"sheet{index + 1}",  # Generate a unique key
                "data": sheet_data  # Preserve full data including headers
            }

            if sheet_data:
                sheet_obj["rows"] = len(sheet_data)
                sheet_obj["columns"] = len(sheet_data[0]) if sheet_data[0] else 0  # Ensure valid column count

            sheets.append(sheet_obj)

        return jsonify(sheets)  # Send JSON response
