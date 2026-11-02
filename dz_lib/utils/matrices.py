from pandas import DataFrame
from io import BytesIO
import pyexcel as p
import numpy as np
import pandas as pd
from dz_lib.univariate.distributions import kde_function, pdp_function
from dz_lib.univariate import metrics

def generate_data_frame(samples, row_labels=None, col_labels=None, metric="similarity", function_type="kde",
                        bandwidth=10):
    """Pairwise comparison matrix for `samples`.

    Similarity, likeness and cross-correlation compare KDEs (with `bandwidth`) or 1-sigma
    PDPs, per `function_type`. KS and Kuiper compare the empirical CDFs of the ages, as the
    tests are defined. A `dis_` prefix returns 1 - value.
    """
    dis = metric.startswith("dis_")
    base = metric[4:] if dis else metric
    if base in ("ks", "kuiper"):
        data = [sample.get_ages() for sample in samples]
        compare = metrics.ks_ages if base == "ks" else metrics.kuiper_ages
    elif base in ("similarity", "likeness", "cross_correlation"):
        if function_type == "kde":
            data = [kde_function(sample, bandwidth).y_values for sample in samples]
        else:
            data = [pdp_function(sample).y_values for sample in samples]
        compare = {"similarity": metrics.similarity, "likeness": metrics.likeness,
                   "cross_correlation": metrics.r2}[base]
    else:
        raise ValueError(f"Unknown metric {metric}")

    num_data_sets = len(samples)
    matrix = np.zeros((num_data_sets, num_data_sets))
    for i in range(num_data_sets):
        for j in range(num_data_sets):
            value = compare(data[i], data[j])
            matrix[i, j] = np.round(1 - value if dis else value, 2)

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
