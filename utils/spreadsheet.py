import openpyxl
import base64
from utils.sample import Sample, Grain
from io import BytesIO


def read_samples(excel_data):
    samples = []
    spreadsheet_content = base64.b64decode(excel_data)
    spreadsheet_array = text_to_array(spreadsheet_content)

    # Assuming the sample data starts from row 2 and each sample occupies two columns
    for row_data in spreadsheet_array[1:]:  # Skip the header row
        sample_name = row_data[0]
        if sample_name is not None:
            grains = []
            for i in range(1, len(row_data), 2):  # Iterate over every other column
                age = row_data[i]
                uncertainty = row_data[i + 1] if i + 1 < len(row_data) else None
                if age is not None and uncertainty is not None:
                    grains.append(Grain(float(age), float(uncertainty)))
            sample = Sample(sample_name, grains)
            samples.append(sample)

    return samples


def is_sample_sheet(self):
    # TODO: check if the file is formatted correctly, and then work this into the read_samples function
    if True:
        return True


def extract_data_from_file(file):
    with open(file, "rb") as file:
        xlsx_data = file.read()
    encoded_xlsx_data = base64.b64encode(xlsx_data).decode("utf-8")
    return encoded_xlsx_data


def text_to_array(spreadsheet_content):
    xlsx_io = BytesIO(spreadsheet_content)
    workbook = openpyxl.load_workbook(xlsx_io)
    sheet = workbook.active
    max_rows = sheet.max_row
    max_cols = sheet.max_column
    spreadsheet_data = []
    for row in range(1, max_rows + 1):
        row_data = []
        for col in range(1, max_cols + 1):
            cell_value = sheet.cell(row=row, column=col).value
            cell_value = cell_value if cell_value is not None else None
            row_data.append(cell_value)
        spreadsheet_data.append(row_data)
    return spreadsheet_data
