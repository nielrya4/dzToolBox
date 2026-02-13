# Julia Subprocess Approach - Clean Architecture

## Problem with Previous Approach

The juliacall/PythonCall interop approach had critical issues:
1. **Performance:** Julia functions hung indefinitely or ran extremely slowly
2. **Complexity:** Managing Julia module scope and function accessibility from Python was fragile
3. **Debugging:** Difficult to debug Julia errors from Python
4. **Maintenance:** Changes required deep understanding of both Julia and Python interop

## New Solution: Call Julia Scripts Directly

Instead of trying to call Julia functions from within Python, we now:

1. **Write Julia scripts** that run standalone
2. **Call them via subprocess** from Python
3. **Exchange data via files** (Excel in, JSON out)
4. **Visualize in Python** using the JSON output

This is simpler, faster, and more maintainable.

---

## Architecture

### Data Flow

```
Python (Celery Task)
    ↓
    1. Write Excel file (row-based format)
    ↓
Julia Script (rank_selection.jl)
    ↓
    2. Transform data → Run analysis → Output JSON
    ↓
Python (Celery Task)
    ↓
    3. Parse JSON → Create visualizations → Save outputs
```

### File Locations

```
dzToolBox/
├── julia_scripts/
│   └── rank_selection.jl       # Standalone Julia script for rank selection
├── celery_tasks.py              # Python calls Julia via subprocess
└── utils/
    └── tensor_factorization.py  # Python visualization functions
```

---

## Julia Script: `rank_selection.jl`

### Purpose
Performs the exact tensor factorization rank selection analysis from the original DZ Grainalyzer.

### Input
- Excel file with row-based format:
  ```
  SINK ID | GRAIN ID | Age | Feature1 | Feature2 | ...
  Sample1 | grain_1  | 115 | 0.24     | 650      | ...
  Sample1 | grain_2  | 120 | 0.23     | 655      | ...
  Sample2 | grain_1  | 118 | 0.25     | 645      | ...
  ```

### Processing Steps

1. **Data Transformation**
   - Converts row-based format to column-based (one sheet per feature)
   - This matches the format expected by original DZ Grainalyzer

2. **KDE Generation**
   - Calculates bandwidths using Scott's rule with alpha=0.9, inner_percentile=95
   - Creates KDEs for each feature in each sample
   - Standardizes KDEs to common domains

3. **Tensor Factorization**
   - Builds 3D tensor from KDE evaluations
   - Normalizes fibers (sum to 1)
   - Runs nnmtf for each rank from min_rank to max_rank
   - Uses: projection=:nnscale, maxiter=6000, tol=1e-5, rescale_Y=false

4. **Rank Selection**
   - Calculates final relative errors for each rank
   - Computes curvature using standard_curvature()
   - Excludes last rank from selection
   - Selects rank with maximum curvature

### Output
JSON file with:
```json
{
  "status": "success",
  "ranks": [2, 3, 4, ..., 11],
  "relative_errors": [0.23, 0.18, 0.15, ...],
  "curvatures": [0.05, 0.12, 0.08, ...],
  "best_rank": 3,
  "r2": 0.85,
  "sink_names": ["Sample1", "Sample2", ...],
  "measurement_names": ["Age", "Eu_anomaly", ...]
}
```

### Usage
```bash
julia rank_selection.jl input.xlsx 2 11 output.json
```

---

## Python Integration: `celery_tasks.py`

### find_optimal_rank_task()

**Modified section (lines 627-700):**

```python
# Create temporary Excel file
temp_excel = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
temp_json = tempfile.NamedTemporaryFile(suffix='.json', delete=False)

# Write data in row-based format
import openpyxl
wb = openpyxl.Workbook()
ws = wb.active
header = ['SINK ID', 'GRAIN ID'] + feature_names
ws.append(header)

for sample in active_samples:
    for grain_idx, grain in enumerate(sample.grains):
        row = [sample.name, f'grain_{grain_idx+1}']
        for feature_name in feature_names:
            row.append(grain.features.get(feature_name, None))
        ws.append(row)

wb.save(temp_excel.name)

# Call Julia script
julia_script = os.path.join(os.path.dirname(__file__), 'julia_scripts', 'rank_selection.jl')
cmd = ['julia', julia_script, temp_excel.name, str(min_rank), str(max_rank), temp_json.name]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

if result.returncode != 0:
    raise RuntimeError(f"Julia script failed: {result.stderr}")

# Parse JSON results
with open(temp_json.name, 'r') as f:
    julia_results = json.load(f)

ranks = julia_results['ranks']
errors = julia_results['relative_errors']
curvatures = julia_results['curvatures']
best_rank = julia_results['best_rank']

# Create visualization with Python
graph_fig = tensor_factorization.visualize_rank_selection(
    ranks=ranks,
    errors=errors,
    curvatures=curvatures,
    best_rank=best_rank,
    ...
)
```

---

## Advantages of This Approach

### 1. Performance
- ✅ No juliacall overhead
- ✅ Julia runs at full native speed
- ✅ No Python-Julia interop bottlenecks
- ✅ Can timeout cleanly (10 minute limit)

### 2. Simplicity
- ✅ Clean separation: Julia does math, Python does visualization
- ✅ No module scope issues
- ✅ No wrapper functions needed
- ✅ Standard subprocess interface

### 3. Debugging
- ✅ Julia errors printed to stderr
- ✅ Can run Julia script independently for testing
- ✅ JSON output is human-readable
- ✅ Temp files kept on error for inspection

### 4. Maintainability
- ✅ Julia code can be modified without touching Python
- ✅ Python code doesn't need Julia expertise
- ✅ Easy to add new Julia scripts for different analyses
- ✅ Version control friendly (separate .jl files)

### 5. Correctness
- ✅ Uses exact original DZ Grainalyzer code
- ✅ No Python reimplementation errors
- ✅ Same results as original implementation
- ✅ Can compare JSON output directly with original

---

## Testing

### Test Julia Script Independently

```bash
# Create test data
julia rank_selection.jl \
    static/global/docs/example_tensor_data.xlsx \
    2 11 \
    test_output.json

# Check results
cat test_output.json | jq '.best_rank'
```

### Test From Python

```python
import subprocess
import json

cmd = ['julia', 'julia_scripts/rank_selection.jl', 'test.xlsx', '2', '11', 'out.json']
result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

with open('out.json') as f:
    results = json.load(f)

print(f"Best rank: {results['best_rank']}")
```

---

## Error Handling

### Julia Errors

If Julia script fails, it writes error info to JSON:

```json
{
  "status": "error",
  "error": "UndefVarError: x not defined",
  "stacktrace": "..."
}
```

Python checks status and raises appropriate error.

### Timeout

Subprocess has 10-minute timeout (600 seconds). If exceeded, Python raises TimeoutError and cleans up temp files.

### Cleanup

Temp files are always cleaned up in finally block, even on error.

---

## Future Extensions

### Additional Analysis Scripts

We can add more Julia scripts for other analyses:

- `tensor_factorization.jl` - Full factorization with chosen rank
- `view_kdes.jl` - Generate KDE visualizations
- `bootstrap_analysis.jl` - Uncertainty quantification
- `factor_interpretation.jl` - Extract factor loadings

Each follows the same pattern:
1. Accept Excel input
2. Run Julia analysis
3. Output JSON results
4. Python visualizes

### Shared Julia Code

Common functions can be moved to a Julia module:

```
julia_scripts/
├── DZToolBoxLib/
│   ├── src/
│   │   ├── DZToolBoxLib.jl
│   │   ├── kde.jl
│   │   └── transform.jl
│   └── Project.toml
├── rank_selection.jl  # Uses DZToolBoxLib
└── tensor_factorization.jl  # Uses DZToolBoxLib
```

---

## Migration from Old Approach

### Files Modified

| File | Change | Lines |
|------|--------|-------|
| `celery_tasks.py` | Call Julia via subprocess | 627-700 |
| `celery_tasks.py` | Add `import os` | 13 |
| `julia_scripts/rank_selection.jl` | NEW | All |

### Files No Longer Needed

- `utils/tensor_factorization.py` - `create_kde_tensor_from_samples()` can be removed
- `utils/tensor_factorization.py` - `initialize_julia_packages()` can be removed
- The juliacall wrapper functions are no longer used

### Visualization Functions Kept

Python visualization functions are still used:
- `visualize_rank_selection()`
- `visualize_factors()`
- `visualize_reconstruction_comparison()`

These now work with data from Julia JSON output instead of Python tensors.

---

## Summary

**Old Approach:**
- Python ↔ juliacall ↔ Julia (slow, complex, fragile)

**New Approach:**
- Python → subprocess → Julia → JSON → Python (fast, simple, robust)

**Result:**
- Exact same numerical results as original DZ Grainalyzer
- Much faster execution
- Easier to maintain and debug
- Clean separation of concerns
