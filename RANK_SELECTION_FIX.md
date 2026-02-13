# Rank Selection Fix: Comparison with Original DZ Grainalyzer

## Problem

Our implementation was selecting rank 12 as optimal when the original DZ Grainalyzer selected rank 3 for the same dataset.

## Root Cause Analysis

After deep-diving into the original DZ Grainalyzer code (`/tmp/dzg/worker/src/helpers.jl`), I identified **3 critical differences** in our rank selection algorithm:

### 1. ❌ **Last Rank Exclusion** (MOST CRITICAL)

**Original Code (lines 146-148):**
```julia
# CRITICAL: Remove last rank from consideration
all_rel_errors = all_rel_errors[1:end-1]
relative_errors = relative_errors[1:end-1]
standard_relative_errors = standard_relative_errors[1:end-1]

best_rank = argmax(standard_relative_errors)
```

**Our Code (BEFORE FIX):**
```python
# We were including ALL ranks in the selection
best_rank_idx = curvatures.index(max(curvatures))
best_rank = ranks[best_rank_idx]
```

**Why This Matters:**
- The last rank(s) in the tested range often have artificially high curvature
- This leads to overfitting and selecting ranks that are too high
- In our case: testing ranks 2-15, selecting rank 12 (near the end)
- In original: testing ranks 2-11, excluding rank 11, selecting from 2-10

### 2. ❌ **Not Using standard_curvature() Function**

**Original Code (line 144):**
```julia
standard_relative_errors = standard_curvature(relative_errors)
```

**Our Code (BEFORE FIX):**
```python
# Manual curvature calculation
curvatures = []
for i in range(len(errors)):
    if i == 0 or i == len(errors) - 1:
        curvatures.append(0.0)
    else:
        curvature = errors[i-1] - 2*errors[i] + errors[i+1]  # No abs()!
        curvatures.append(curvature)
```

**Why This Matters:**
- The existing `standard_curvature()` function uses `abs()` to get magnitude
- Without `abs()`, negative curvatures could interfere with selection
- Reinventing the wheel when we already have the correct function

### 3. ❌ **Missing Absolute Value in Curvature**

**standard_curvature() function (tensor_factorization.py line 1399):**
```python
d2 = errors[i-1] - 2*errors[i] + errors[i+1]
curvatures.append(abs(d2))  # ✓ Uses absolute value
```

**Our manual calculation (BEFORE FIX):**
```python
curvature = errors[i-1] - 2*errors[i] + errors[i+1]  # ❌ No abs()
curvatures.append(curvature)
```

## Solution Implemented

### Changes to `celery_tasks.py` (lines 685-722)

```python
# Convert R² to errors (misfit = 1 - R²)
errors = [1 - r2 for r2 in r2_values]

# Calculate curvatures using standard_curvature function (same as original dzgrainalyzer)
curvatures = tensor_factorization.standard_curvature(errors)

# CRITICAL: Exclude the last rank from consideration (same as original dzgrainalyzer)
# The last rank often has artificially high curvature and may overfit
if len(ranks) > 1:
    ranks_for_selection = ranks[:-1]
    errors_for_selection = errors[:-1]
    curvatures_for_selection = curvatures[:-1]
else:
    ranks_for_selection = ranks
    errors_for_selection = errors
    curvatures_for_selection = curvatures

# Find best rank (maximum curvature indicates elbow point)
best_rank_idx = curvatures_for_selection.index(max(curvatures_for_selection))
best_rank = ranks_for_selection[best_rank_idx]

print(f"Rank selection: Best rank is {best_rank} (max curvature = {max(curvatures_for_selection):.6f})")

# Generate rank selection visualization (use full arrays for visualization)
graph_fig = tensor_factorization.visualize_rank_selection(
    ranks=ranks,
    errors=errors,
    curvatures=curvatures,
    best_rank=best_rank,
    # ... other params
)
```

## What the Fix Does

1. **Uses the existing `standard_curvature()` function** instead of manual calculation
2. **Excludes the last rank** from the selection process (ranks[:-1])
3. **Maintains consistency** with the original DZ Grainalyzer algorithm
4. **Visualizes all ranks** but only selects from ranks 2 to (max_rank - 1)

## Expected Behavior After Fix

- **Before:** Testing ranks 2-15 → Selecting from all 14 ranks → Often picks rank 12-13
- **After:** Testing ranks 2-15 → Selecting from ranks 2-14 (excludes 15) → Should pick lower, more appropriate rank

For your dataset:
- **Original DZ Grainalyzer:** Rank 3 (optimal)
- **Our Implementation (BEFORE):** Rank 12 (overfitting)
- **Our Implementation (AFTER):** Should select rank 3 or similar

## Verification Steps

1. ✅ Celery worker restarted with fixed code
2. 🔄 Rerun "Find Optimal Rank" analysis on the same dataset
3. 🔍 Check that selected rank is lower and closer to the original (rank 3)
4. 📊 Verify the rank selection plot shows the correct optimal rank

## Other Implementation Details Verified as Correct

✅ **Factorization Algorithm:** Using `nnmtf` with correct parameters
```python
projection=:nnscale,
maxiter=6000,
tol=1e-5,
rescale_Y=false
```

✅ **Fiber Normalization:** Pre-normalizing before factorization (line 377)
```python
jl_tensor_normalized = jl.normalize_fibers(jl_tensor)
```

✅ **Error Metric:** Using relative reconstruction error (1 - R²)

✅ **Curvature Function:** `standard_curvature()` implementation matches expected behavior

## References

- Original DZ Grainalyzer: `/tmp/dzg/worker/src/helpers.jl` lines 99-218
- Our Implementation: `celery_tasks.py` lines 571-787
- Curvature Function: `utils/tensor_factorization.py` lines 1373-1402
- Visualization: `utils/tensor_factorization.py` lines 1141-1219

## Testing Recommendation

Run the "Find Optimal Rank" analysis with:
- Min Rank: 2
- Max Rank: 15 (or 10 to match original more closely)
- Same dataset as before

Expected: Should now select rank 3 (or close to it) instead of rank 12.
