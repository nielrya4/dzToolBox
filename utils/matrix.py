from utils import test, graph
import numpy as np
import pandas as pd
from io import BytesIO



class Matrix:
    def __init__(self, samples, matrix_type):
        self.samples = samples
        self.matrix_type = matrix_type

    def generate_data_frame(self, row_labels=None, col_labels=None, matrix_type="similarity"):
        samples = self.samples
        sample_kdes = [graph.kde_function(sample)[1] for sample in samples]
        sample_cdfs = [graph.cdf_function(sample)[1] for sample in samples]
        num_data_sets = len(samples)
        matrix = np.zeros((num_data_sets, num_data_sets))
        if matrix_type == "similarity":
            for i, sample1 in enumerate(sample_kdes):
                for j, sample2 in enumerate(sample_kdes):
                    similarity_score = test.similarity(sample1, sample2)
                    matrix[i, j] = similarity_score
        elif matrix_type == "dissimilarity":
            for i, sample1 in enumerate(sample_kdes):
                for j, sample2 in enumerate(sample_kdes):
                    dissimilarity_score = test.dis_similarity(sample1, sample2)
                    matrix[i, j] = dissimilarity_score
        elif matrix_type == "likeness":
            for i, sample1 in enumerate(sample_kdes):
                for j, sample2 in enumerate(sample_kdes):
                    likeness_score = test.likeness(sample1, sample2)
                    matrix[i, j] = likeness_score
        elif matrix_type == "ks":
            for i, sample1 in enumerate(sample_cdfs):
                for j, sample2 in enumerate(sample_cdfs):
                    ks_score = test.ks(sample1, sample2)
                    matrix[i, j] = ks_score
        elif matrix_type == "kuiper":
            for i, sample1 in enumerate(sample_cdfs):
                for j, sample2 in enumerate(sample_cdfs):
                    kuiper_score = test.kuiper(sample1, sample2)
                    matrix[i, j] = kuiper_score
        elif matrix_type == "r2":
            for i, sample1 in enumerate(sample_kdes):
                for j, sample2 in enumerate(sample_kdes):
                    cross_correlation_score = test.r2(sample1, sample2)
                    matrix[i, j] = cross_correlation_score

        # Create a DataFrame with the normalized similarity scores and labels
        if row_labels is None:
            row_labels = [sample.name for sample in samples]
        if col_labels is None:
            col_labels = [sample.name for sample in samples]

        df = pd.DataFrame(matrix, columns=col_labels, index=row_labels)
        return df

    def to_html(self):
        data_frame = self.generate_data_frame(matrix_type=self.matrix_type)
        html_data = data_frame.to_html(classes="table table-bordered table-striped", justify="center").replace('<th>','<th style = "background-color: White;">').replace('<td>','<td style = "background-color: White;">')
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
