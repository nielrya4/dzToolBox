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


def excel_to_row_array(file_path):
    """
    Read Excel file as row-based array (for multivariate data).
    Unlike excel_to_array(), this does NOT transpose the data.

    Returns:
        List[List]: 2D array where each inner list is a row from Excel
        Row 0 is typically the header row
    """
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        sheet = workbook.active

        spreadsheet_data = []
        for row in sheet.values:
            if row:  # Skip empty rows
                row_data = [cell if cell is not None else None for cell in row]
                spreadsheet_data.append(row_data)

        workbook.close()

        if not spreadsheet_data:
            return None

        return spreadsheet_data
    except Exception as e:
        print(f"Error reading Excel file: {e}")
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


def read_multivariate_samples(
    spreadsheet_array,
    sink_id_col=0,
    grain_id_col=1,
    feature_start_col=2,
    max_age=4500
):
    """
    Read row-based multivariate grain data from a spreadsheet array.

    Expected Format:
        Row 0: ['SINK ID', 'GRAIN ID', 'Age', 'Feature1', 'Feature2', ...]
        Row 1: ['Sample1', 'Grain1', 116.74, 0.242, 617.85, ...]
        Row 2: ['Sample1', 'Grain2', 138.1, 0.298, 621.56, ...]
        ...

    Parameters:
        spreadsheet_array: 2D array from spreadsheet (list of rows)
        sink_id_col: Column index for sample names (default 0)
        grain_id_col: Column index for grain IDs (default 1)
        feature_start_col: First column with feature data (default 2)
        max_age: Optional age filter for 'Age' feature (default 4500)

    Returns:
        Tuple of:
            - List[MultivariateSample]: Samples grouped by SINK_ID
            - List[str]: Feature names in order (from header row)

    Raises:
        ValueError: If data format is invalid or insufficient samples
    """
    from utils.multivariate_sample import MultivariateGrain, MultivariateSample
    from collections import defaultdict

    if not spreadsheet_array or len(spreadsheet_array) < 2:
        raise ValueError("Invalid data format: Need at least header row and one data row")

    # Extract header row and feature names
    header_row = spreadsheet_array[0]
    if len(header_row) <= feature_start_col:
        raise ValueError(f"Invalid data format: Expected at least {feature_start_col + 1} columns")

    feature_names = [str(name).strip() for name in header_row[feature_start_col:] if name is not None]

    if not feature_names:
        raise ValueError("No feature columns found in header")

    # Validate that 'Age' is present (optional but recommended)
    if 'Age' not in feature_names:
        print("Warning: 'Age' feature not found in data. This may affect some analyses.")

    # Group grains by SINK_ID
    sample_grains = defaultdict(list)
    age_feature_index = feature_names.index('Age') if 'Age' in feature_names else None

    skipped_rows = 0
    for row_idx, row in enumerate(spreadsheet_array[1:], start=1):
        if not row or len(row) <= feature_start_col:
            continue

        # Extract SINK_ID and GRAIN_ID
        sink_id = row[sink_id_col]
        grain_id = row[grain_id_col]

        if sink_id is None or grain_id is None:
            skipped_rows += 1
            continue

        # Extract feature values
        feature_values = row[feature_start_col:feature_start_col + len(feature_names)]

        # Build feature dictionary
        features = {}
        skip_grain = False
        for feat_name, feat_value in zip(feature_names, feature_values):
            # Check for missing or non-numeric values
            if feat_value is None:
                skip_grain = True
                break

            try:
                features[feat_name] = float(feat_value)
            except (ValueError, TypeError):
                skip_grain = True
                break

        if skip_grain:
            skipped_rows += 1
            continue

        # Apply age filter if Age feature exists
        if age_feature_index is not None:
            age_value = features['Age']
            if age_value > max_age:
                skipped_rows += 1
                continue

        # Create grain and add to sample
        grain = MultivariateGrain(grain_id=str(grain_id), features=features)
        sample_grains[str(sink_id)].append(grain)

    if skipped_rows > 0:
        print(f"Info: Skipped {skipped_rows} rows due to missing/invalid values or age filter")

    # Create MultivariateSample objects
    samples = []
    for sink_id, grains in sample_grains.items():
        if len(grains) < 10:
            print(f"Warning: Sample '{sink_id}' has only {len(grains)} grains (minimum 10 recommended)")

        sample = MultivariateSample(name=sink_id, grains=grains, feature_names=feature_names)
        samples.append(sample)

    # Validate minimum sample count
    if len(samples) < 2:
        raise ValueError(f"At least 2 samples required for factorization, got {len(samples)}")

    return samples, feature_names
