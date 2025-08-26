import openpyxl
from utils.sample import Sample, Grain
import json
import orjson  # Much faster JSON library


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
        # Use read_only mode for better memory efficiency
        workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        sheet = workbook.active
        
        # Read all data at once using sheet.values (much faster)
        spreadsheet_data = []
        for row in sheet.values:
            if row:  # Skip empty rows
                # Convert None values and ensure consistent row length
                row_data = [cell if cell is not None else None for cell in row]
                spreadsheet_data.append(row_data)
        
        workbook.close()  # Explicitly close to free memory
        
        if not spreadsheet_data:
            return None
            
        # Transpose more efficiently using list comprehension with proper bounds checking
        max_cols = max(len(row) for row in spreadsheet_data) if spreadsheet_data else 0
        transposed_array = []
        for i in range(max_cols):
            col = [row[i] if i < len(row) else None for row in spreadsheet_data]
            transposed_array.append(col)
        
        return transposed_array
    except Exception as e:
        print(f"Error converting Excel file to array: {e}")
        return None


def array_to_text(array):
    try:
        # Use orjson for faster serialization (fallback to json if not available)
        try:
            return orjson.dumps(array).decode('utf-8')
        except NameError:
            return json.dumps(array)
    except Exception as e:
        print(f"Error converting array to text: {e}")
        return None


def text_to_array(text):
    try:
        # Use orjson for faster deserialization (fallback to json if not available)
        try:
            return orjson.loads(text.encode('utf-8') if isinstance(text, str) else text)
        except NameError:
            return json.loads(text)
    except Exception as e:
        print(f"Error converting text to array: {e}")
        return None
