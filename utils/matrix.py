from utils import test
import numpy as np
import pandas as pd
from io import BytesIO


class Matrix:
    def __init__(self, samples, matrix_type):
        self.samples = samples
        self.matrix_type = matrix_type

    def generate_data_frame(self, row_labels=None, col_labels=None, matrix_type="similarity"):
        samples = self.samples
        num_data_sets = len(samples)
        matrix = np.zeros((num_data_sets, num_data_sets))
        if matrix_type == "similarity":
            for i, sample1 in enumerate(samples):
                for j, sample2 in enumerate(samples):
                    similarity_score = test.similarity(sample1, sample2)
                    matrix[i, j] = similarity_score
        elif matrix_type == "dissimilarity":
            for i, sample1 in enumerate(samples):
                for j, sample2 in enumerate(samples):
                    dissimilarity_score = test.dis_similarity(sample1, sample2)
                    matrix[i, j] = dissimilarity_score
        elif matrix_type == "likeness":
            for i, sample1 in enumerate(samples):
                for j, sample2 in enumerate(samples):
                    likeness_score = test.likeness(sample1, sample2)
                    matrix[i, j] = likeness_score
        elif matrix_type == "ks":
            for i, sample1 in enumerate(samples):
                for j, sample2 in enumerate(samples):
                    ks_score = test.ks(sample1, sample2)
                    matrix[i, j] = ks_score
        elif matrix_type == "kuiper":
            for i, sample1 in enumerate(samples):
                for j, sample2 in enumerate(samples):
                    kuiper_score = test.kuiper(sample1, sample2)
                    matrix[i, j] = kuiper_score
        elif matrix_type == "r2":
            for i, sample1 in enumerate(samples):
                for j, sample2 in enumerate(samples):
                    cross_correlation_score = test.r2(sample1, sample2)
                    matrix[i, j] = cross_correlation_score

        # Create a DataFrame with the normalized similarity scores and labels
        if row_labels is None:
            row_labels = [f'Data {i+1}' for i in range(num_data_sets)]
        if col_labels is None:
            col_labels = [f'Data {i+1}' for i in range(num_data_sets)]

        df = pd.DataFrame(matrix, columns=col_labels, index=row_labels)
        return df

    def to_html(self):
        html_data = self.generate_data_frame(matrix_type=self.matrix_type)
        html_data.to_html(classes="table table-bordered table-striped", justify="center").replace('<th>','<th style = "background-color: White;">').replace('<td>','<td style = "background-color: White;">')
        return html_data

    def to_xlsx(self):
        buffer = BytesIO()
        xlsx_data = self.generate_data_frame(matrix_type=self.matrix_type)
        xlsx_data.to_excel(buffer, index=True, engine='openpyxl', header=True)
        buffer.seek(0)
        return buffer

    def to_xls(self):
        buffer = BytesIO()
        xls_data = self.generate_data_frame(matrix_type=self.matrix_type)
        xls_data.to_excel(buffer, index=True, engine='xlwt', header=True)
        buffer.seek(0)
        return buffer

    def to_csv(self):
        buffer = BytesIO()
        csv_data = self.generate_data_frame(matrix_type=self.matrix_type)
        csv_data.to_csv(buffer, index=True, header=True)
        buffer.seek(0)
        return buffer

    def to_json(self):
        json_data = self.generate_data_frame(matrix_type=self.matrix_type)
        json_data.to_json()
        return json_data
