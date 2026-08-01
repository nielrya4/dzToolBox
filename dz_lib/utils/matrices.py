from pandas import DataFrame
from io import BytesIO
import pyexcel as p
import numpy as np
import pandas as pd
from dz_lib.univariate.distributions import kde_function, pdp_function, cdf_function
from dz_lib.univariate import metrics

def generate_data_frame(samples, row_labels=None, col_labels=None, metric="similarity", function_type="kde"):
    if metric == "ks" or metric == "kuiper":
        samples.reverse()
    sample_kdes = [kde_function(sample, 10).y_values for sample in samples]
    sample_pdps = [pdp_function(sample).y_values for sample in samples]
    sample_cdfs = [cdf_function(kde_function(sample, 10)).y_values for sample in samples]
    num_data_sets = len(samples)
    matrix = np.zeros((num_data_sets, num_data_sets))
    if metric == "similarity":
        for i, sample1 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
            for j, sample2 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
                similarity_score = metrics.similarity(sample1, sample2)
                matrix[i, j] = np.round(similarity_score, 2)
    elif metric == "dis_similarity":
        for i, sample1 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
            for j, sample2 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
                dissimilarity_score = metrics.dis_similarity(sample1, sample2)
                matrix[i, j] = np.round(dissimilarity_score, 2)
    elif metric == "likeness":
        for i, sample1 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
            for j, sample2 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
                likeness_score = metrics.likeness(sample1, sample2)
                matrix[i, j] = np.round(likeness_score, 2)
    elif metric == "dis_likeness":
        for i, sample1 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
            for j, sample2 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
                likeness_score = metrics.dis_likeness(sample1, sample2)
                matrix[i, j] = np.round(likeness_score, 2)
    elif metric == "ks":
        for i, sample1 in enumerate(sample_cdfs):
            for j, sample2 in enumerate(sample_cdfs):
                ks_score = metrics.ks(sample1, sample2)
                matrix[i, j] = np.round(ks_score, 2)
    elif metric == "dis_ks":
        for i, sample1 in enumerate(sample_cdfs):
            for j, sample2 in enumerate(sample_cdfs):
                ks_score = metrics.dis_ks(sample1, sample2)
                matrix[i, j] = np.round(ks_score, 2)
    elif metric == "kuiper":
        for i, sample1 in enumerate(sample_cdfs):
            for j, sample2 in enumerate(sample_cdfs):
                kuiper_score = metrics.kuiper(sample1, sample2)
                matrix[i, j] = np.round(kuiper_score, 2)
    elif metric == "dis_kuiper":
        for i, sample1 in enumerate(sample_cdfs):
            for j, sample2 in enumerate(sample_cdfs):
                kuiper_score = metrics.dis_kuiper(sample1, sample2)
                matrix[i, j] = np.round(kuiper_score, 2)
    elif metric == "cross_correlation":
        for i, sample1 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
            for j, sample2 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
                cross_correlation_score = metrics.r2(sample1, sample2)
                matrix[i, j] = np.round(cross_correlation_score, 2)
    elif metric == "dis_cross_correlation":
        for i, sample1 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
            for j, sample2 in enumerate(sample_kdes if function_type == "kde" else sample_pdps):
                cross_correlation_score = metrics.dis_r2(sample1, sample2)
                matrix[i, j] = np.round(cross_correlation_score, 2)
    else:
        raise ValueError(f"Unknown metric {metric}")
    if row_labels is None:
        row_labels = [sample.name for sample in samples]
    if col_labels is None:
        col_labels = [sample.name for sample in samples]

    df = pd.DataFrame(matrix, columns=col_labels, index=row_labels)
    return df

def dataframe_to_html(df: DataFrame, title:str=None):
    return (
        (f"<h4>{title}</h4>" if title else "") +
        df.to_html(
            classes="table table-bordered table-striped",
            justify="center",
            index=True,
        ).replace(
            '<th>',
            '<th style="background-color: White;">'
        ).replace(
            '<td>',
            '<td style="background-color: White;">'
        )
    )

def to_xlsx(data_frame: DataFrame, include_header: bool=True, include_index: bool=True):
    buffer = BytesIO()
    data_frame.to_excel(buffer, index=include_index, engine='openpyxl', header=include_header)
    buffer.seek(0)
    return buffer


def to_xls(data_frame: DataFrame, include_header: bool = True, include_index: bool = True):
    buffer = BytesIO()

    df = data_frame.copy()  # Copy to avoid modifying the original DataFrame

    records = df.reset_index().values.tolist() if include_index else df.values.tolist()

    if include_header:
        headers = df.columns.tolist()
        if include_index:
            headers.insert(0, df.index.name if df.index.name else "")  # Add index name or default "Index"
        records.insert(0, headers)  # Insert headers as the first row

    p.save_as(array=records, dest_file_type='xls', dest_file_stream=buffer)
    buffer.seek(0)

    return buffer

def to_csv(data_frame: DataFrame, include_header: bool=True, include_index: bool=True):
    buffer = BytesIO()
    data_frame.to_csv(buffer, index=include_index, header=include_header)
    buffer.seek(0)
    return buffer

def to_json(data_frame: DataFrame):
    return data_frame.to_json()
