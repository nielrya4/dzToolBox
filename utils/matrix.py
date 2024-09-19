from utils import test, graph
import numpy as np
import pandas as pd
from io import BytesIO
import base64
import pyexcel as p


class Matrix:
    def __init__(self, samples, matrix_type, function_type="kde"):
        self.samples = samples
        self.matrix_type = matrix_type
        self.function_type = function_type
        self.data_frame = self.generate_data_frame(matrix_type=matrix_type, function_type=function_type)

    def generate_data_frame(self, row_labels=None, col_labels=None, matrix_type="similarity", function_type="kde"):
        samples = self.samples
        if matrix_type == "ks" or matrix_type == "kuiper":
            samples.reverse()
        sample_kdes = [graph.kde_function(sample)[1] for sample in samples]
        sample_pdps = [graph.pdp_function(sample)[1] for sample in samples]
        sample_cdfs = [graph.cdf_function(sample)[1] for sample in samples]
        num_data_sets = len(samples)
        matrix = np.zeros((num_data_sets, num_data_sets))
        if matrix_type == "similarity":
            for i, sample1 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
                for j, sample2 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
                    similarity_score = test.similarity(sample1, sample2)
                    matrix[i, j] = np.round(similarity_score, 2)
        elif matrix_type == "dissimilarity":
            for i, sample1 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
                for j, sample2 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
                    dissimilarity_score = test.dis_similarity(sample1, sample2)
                    matrix[i, j] = np.round(dissimilarity_score, 2)
        elif matrix_type == "likeness":
            for i, sample1 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
                for j, sample2 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
                    likeness_score = test.likeness(sample1, sample2)
                    matrix[i, j] = np.round(likeness_score, 2)
        elif matrix_type == "ks":
            for i, sample1 in enumerate(sample_cdfs):
                for j, sample2 in enumerate(sample_cdfs):
                    ks_score = test.ks(sample1, sample2)
                    matrix[i, j] = np.round(ks_score, 2)
        elif matrix_type == "kuiper":
            for i, sample1 in enumerate(sample_cdfs):
                for j, sample2 in enumerate(sample_cdfs):
                    kuiper_score = test.kuiper(sample1, sample2)
                    matrix[i, j] = np.round(kuiper_score, 2)
        elif matrix_type == "r2":
            for i, sample1 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
                for j, sample2 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
                    cross_correlation_score = test.r2(sample1, sample2)
                    matrix[i, j] = np.round(cross_correlation_score, 2)

        if row_labels is None:
            row_labels = [sample.name for sample in samples]
        if col_labels is None:
            col_labels = [sample.name for sample in samples]

        df = pd.DataFrame(matrix, columns=col_labels, index=row_labels)
        return df

    def to_html(self, output_id, actions_button=False):
        data_frame = self.data_frame
        xlsx_data = self.to_xlsx()
        xls_data = self.to_xls()
        csv_data = self.to_csv()

        encoded_xlsx_data = base64.b64encode(xlsx_data.getvalue()).decode('utf-8')
        encoded_xls_data = base64.b64encode(xls_data.getvalue()).decode('utf-8')
        encoded_csv_data = base64.b64encode(csv_data.getvalue()).decode('utf-8')


        html_data = "<div>"
        html_data += data_frame.to_html(classes="table table-bordered table-striped", justify="center").replace('<th>','<th style = "background-color: White;">').replace('<td>','<td style = "background-color: White;">')
        if actions_button:
            html_data += f"""<div class="dropdown show">
                                <a class="btn btn-secondary dropdown-toggle" href="#" role="button" id="{output_id}_dropdown" data-bs-toggle="dropdown" aria-expanded="false">
                                    Actions
                                </a>
                                <div class="dropdown-menu" aria-labelledby="{output_id}_dropdown">
                                    <a class="dropdown-item" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{encoded_xlsx_data}" download="file.xlsx">Download As XLSX</a>
                                    <a class="dropdown-item" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{encoded_xls_data}" download="file.xls">Download As XLS</a>
                                    <a class="dropdown-item" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{encoded_csv_data}" download="file.csv">Download As CSV</a>
                                    <a class="dropdown-item" href="#" data-hx-post="/delete_output/{output_id}" data-hx-target="#outputs_container" data-hx-swap="innerHTML" onclick="show_delete_output_spinner();">Delete This Output</a>
                                </div>
                            </div>"""
        html_data += "</div>"
        return html_data

    def to_xlsx(self):
        buffer = BytesIO()
        xlsx_data = self.data_frame
        xlsx_data.to_excel(buffer, index=True, engine='openpyxl', header=True)
        buffer.seek(0)
        return buffer

    def to_xls(self):
        buffer = BytesIO()
        df = self.data_frame
        records = df.reset_index().values.tolist()
        p.save_as(array=records, dest_file_type='xls', dest_file_stream=buffer)
        buffer.seek(0)
        return buffer

    def to_csv(self):
        buffer = BytesIO()
        csv_data = self.data_frame
        csv_data.to_csv(buffer, index=True, header=True)
        buffer.seek(0)
        return buffer

    def to_json(self):
        json_data = self.data_frame
        json_data.to_json()
        return json_data
