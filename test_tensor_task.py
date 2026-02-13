"""
Test script to debug tensor factorization task issues
"""
import sys
from utils import spreadsheet

# Simulate what happens when data is uploaded
print("=" * 60)
print("Testing data upload/storage/retrieval cycle")
print("=" * 60)

# Step 1: Read Excel file (simulating upload)
print("\n1. Reading Excel file...")
data_from_excel = spreadsheet.excel_to_array('static/global/docs/example_tensor_data.xlsx')
print(f"   excel_to_array result: {len(data_from_excel)} items")
print(f"   First item length: {len(data_from_excel[0])}")
print(f"   First few elements: {data_from_excel[0][:5]}")

# Step 2: Transpose (as done in project_browser.py line 39)
import numpy as np
transposed = np.transpose(data_from_excel)
transposed_list = np.ndarray.tolist(transposed)
print(f"\n2. After transpose (as stored in DB):")
print(f"   Shape: {len(transposed_list)} rows")
print(f"   First row length: {len(transposed_list[0])}")
print(f"   First row: {transposed_list[0][:5]}")

# Step 3: Convert to text (as stored)
text_data = spreadsheet.array_to_text(transposed_list)
print(f"\n3. Converted to text: {len(text_data)} characters")

# Step 4: Retrieve from text (simulating celery task)
print("\n4. Retrieving from 'database' (text_to_array)...")
retrieved_data = spreadsheet.text_to_array(text_data)
print(f"   Retrieved: {len(retrieved_data)} rows")
print(f"   First row length: {len(retrieved_data[0])}")
print(f"   First row: {retrieved_data[0][:5]}")

# Step 5: Try to read as multivariate
print("\n5. Attempting read_multivariate_samples...")
try:
    samples, features = spreadsheet.read_multivariate_samples(
        spreadsheet_array=retrieved_data,
        max_age=4500
    )
    print(f"   ✓ SUCCESS: {len(samples)} samples, {len(features)} features")
    print(f"   Features: {features}")
    print(f"   First sample: {samples[0].name}, {len(samples[0].grains)} grains")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnosis:")
print("=" * 60)
print("The data format after upload/storage/retrieval is:")
print(f"  - {len(retrieved_data)} rows")
print(f"  - First row has {len(retrieved_data[0])} elements")
print(f"  - Format appears to be: {'ROW' if retrieved_data[0][0] == 'SINK ID' else 'COLUMN'} based")
