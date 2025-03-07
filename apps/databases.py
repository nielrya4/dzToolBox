import pandas as pd
from flask import jsonify
from flask_login import login_required


def register(app):
    @app.route('/databases/<string:file>', methods=['GET'])
    @login_required
    def load_databases(file):
        df_sheets = pd.read_excel(f"static/databases/{file}.xlsx", sheet_name=None, header=None)

        # Extracting the first sheet only for Handsontable
        first_sheet_name, first_df = next(iter(df_sheets.items()))
        first_df = first_df.dropna(how="all").fillna("").reset_index(drop=True)

        return jsonify(first_df.values.tolist())
