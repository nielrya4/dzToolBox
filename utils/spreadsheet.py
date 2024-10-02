import openpyxl
from utils.sample import Sample, Grain
import json


def read_samples(spreadsheet_array):
    samples = []
    for i in range(0, len(spreadsheet_array[0]), 2):
        sample_name = spreadsheet_array[0][i]
        if sample_name is not None:
            grains = []
            for row_data in spreadsheet_array[1:]:
                age = row_data[i]
                if not (isinstance(age, float) or isinstance(age, int)):
                    age = None
                uncertainty = row_data[i + 1] if i + 1 < len(row_data) else None
                if not (isinstance(uncertainty, float) or isinstance(uncertainty, int)):
                    uncertainty = None
                if age is not None and uncertainty is not None and float(age) < 4500: # TODO: make min and max grains a project setting.
                    grains.append(Grain(float(age), float(uncertainty)))
            sample = Sample(sample_name, grains)
            samples.append(sample)
    return samples


def is_sample_sheet(self):
    # TODO: check if the file is formatted correctly, and then work this into the read_samples function
    if True:
        return True


def excel_to_array(file_path):
    try:
        workbook = openpyxl.load_workbook(file_path)
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
        transposed_array = [[spreadsheet_data[j][i] for j in range(len(spreadsheet_data))] for i in range(len(spreadsheet_data[0]))]
        return transposed_array
    except Exception as e:
        print(f"Error converting Excel file to array: {e}")
        return None


def array_to_text(array):
    try:
        # Serialize the array to a string using JSON
        return json.dumps(array)
    except Exception as e:
        print(f"Error converting array to text: {e}")
        return None


def text_to_array(text):
    try:
        array = json.loads(text)
        return array
    except Exception as e:
        print(f"Error converting text to array: {e}")
        return None
