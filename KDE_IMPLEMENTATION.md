# KDE Implementation: Matching Original DZ Grainalyzer

## ⚠️ UPDATED: Now Using Julia's Exact KDE Functions

**Latest Change (2026-02-11):** We now call Julia's `KernelDensity.jl` functions directly instead of reimplementing with scipy. See `JULIA_KDE_IMPLEMENTATION.md` for details.

This ensures **exact numerical matching** with the original, not just "close enough" results.

---

## Overview (Original scipy Implementation)

~~We've completely reimplemented the tensor factorization to use KDE (Kernel Density Estimation) instead of raw feature values, **exactly matching the original DZ Grainalyzer approach**.~~

**NOTE:** The scipy implementation below selected the correct rank (3) but produced slightly different error curves. We've now switched to calling Julia's exact functions for perfect matching.

## Changes Made

### 1. New Function: `create_kde_tensor_from_samples()`
**File:** `utils/tensor_factorization.py` (lines 38-177)

Implements the full KDE pipeline that matches the original:

#### Step 1: Calculate Default Bandwidth
```python
# Matches: bandwidths = default_bandwidth.(collect(eachmeasurement(sink1)), alpha_, inner_percentile)
```
- Uses first sample as reference
- Filters to inner_percentile (default 95%)
- Applies Scott's rule: `bandwidth = alpha * std * n^(-1/5)`
- Parameters: `alpha=0.9`, `inner_percentile=95.0`

#### Step 2: Generate KDEs
```python
# Matches: raw_densities = make_densities.(sinks; bandwidths, inner_percentile)
```
- Creates Gaussian KDE for each feature in each sample
- Uses calculated bandwidth
- Handles samples with insufficient data

#### Step 3: Standardize to Common Domains
```python
# Matches: densities, domains = standardize_KDEs(raw_densities)
```
- Determines global domain for each feature (95th percentile range)
- Creates 150 evaluation points per feature
- All samples share the same domain per feature

#### Step 4: Build Tensor from KDE Evaluations
```python
# tensor shape: (n_samples, n_kde_points, n_features)
```
- Evaluates each KDE on its common domain
- Normalizes each density to integrate to 1
- Result: 3D tensor of probability densities

**Tensor Shape Comparison:**
- **Original DZ Grainalyzer:** `(20 samples, ~150 KDE points, 7 features)`
- **Our KDE Implementation:** `(20 samples, 150 KDE points, 7 features)` ✓ **MATCHES**

### 2. Updated `find_optimal_rank_task()`
**File:** `celery_tasks.py`

#### Change 1: Use KDE Tensor (lines 627-639)
```python
# OLD: Create raw tensor
tensor, metadata = tensor_factorization.create_tensor_from_multivariate_samples(...)

# NEW: Create KDE tensor
tensor, metadata = tensor_factorization.create_kde_tensor_from_samples(
    samples=active_samples,
    feature_names=feature_names,
    inner_percentile=95.0,  # Matches original
    alpha=0.9,              # Matches original
    n_kde_points=150        # Matches original
)
```

#### Change 2: Skip Additional Normalization (lines 641-648)
```python
# OLD: Apply minmax normalization
normalized_tensor, norm_params = tensor_factorization.normalize_tensor(tensor, method='minmax', ...)

# NEW: Use KDE tensor directly (already normalized)
normalized_tensor = tensor
norm_params = None  # No denormalization needed
```

**Rationale:** KDEs are already normalized probability densities. The original only does fiber normalization (sum to 1) inside nnmtf, not minmax scaling.

#### Change 3: No Denormalization (line 671)
```python
# OLD: Denormalize reconstruction
denormalized_reconstruction = tensor_factorization.denormalize_tensor(...)

# NEW: Use reconstruction directly
reconstruction = factorization_result['reconstruction']
```

### 3. Updated Default Rank Range
**File:** `templates/editor/modals/find_optimal_rank_modal.html` (line 28)

```html
<!-- OLD -->
<input type="number" id="find_optimal_rank_max" value="15">

<!-- NEW -->
<input type="number" id="find_optimal_rank_max" value="11">
```

**Rationale:** Original DZ Grainalyzer tests ranks up to `min(n_samples, 10)+1 = 11` for 20 samples.

---

## Exact Parameter Matching

| Parameter | Original | Our Implementation | Status |
|-----------|----------|-------------------|--------|
| **Data Type** | KDE densities | KDE densities | ✅ MATCHES |
| **Tensor Shape** | (20, ~150, 7) | (20, 150, 7) | ✅ MATCHES |
| **inner_percentile** | 95 | 95.0 | ✅ MATCHES |
| **alpha** | 0.9 | 0.9 | ✅ MATCHES |
| **n_kde_points** | ~150 | 150 | ✅ MATCHES |
| **Algorithm** | nnmtf | nnmtf | ✅ MATCHES |
| **projection** | :nnscale | :nnscale | ✅ MATCHES |
| **maxiter** | 6000 | 6000 | ✅ MATCHES |
| **tol** | 1e-5 | 1e-5 | ✅ MATCHES |
| **rescale_Y** | false | false | ✅ MATCHES |
| **Fiber normalization** | sum to 1 | sum to 1 | ✅ MATCHES |
| **Rank range (default)** | 2-11 | 2-11 | ✅ MATCHES |
| **Exclude last rank** | Yes | Yes | ✅ MATCHES |
| **Curvature method** | standard_curvature | standard_curvature | ✅ MATCHES |

---

## Testing & Verification

### Expected Results

With the KDE implementation, testing on the same dataset should now give:

- **Original DZ Grainalyzer:** Rank 3
- **Our Implementation (Raw Features):** Rank 10 ❌
- **Our Implementation (KDE):** Rank 3 ✅ **SHOULD MATCH**

### Test Steps

1. ✅ Celery worker restarted with KDE implementation
2. 🔄 Run "Find Optimal Rank" with default settings (ranks 2-11)
3. 🔍 Verify optimal rank matches original (~rank 3)
4. 📊 Compare misfit curves

### Why This Should Match

1. **Same Data Type:** Both use KDE probability densities
2. **Same Bandwidth:** Scott's rule with alpha=0.9, inner_percentile=95%
3. **Same Domain:** 95th percentile range, 150 evaluation points
4. **Same Normalization:** Each KDE integrates to 1, then fibers sum to 1
5. **Same Algorithm:** nnmtf with identical parameters
6. **Same Rank Selection:** standard_curvature, exclude last rank

---

## Key Insights

### Why Raw Features Gave Different Results

- **Raw features:** Discrete measurements at variable scales
  - Age: 100-200 Ma
  - Eu_anomaly: 0.2-0.3
  - Ti_temp: 600-700°C
  - Scale differences dominate the factorization

- **KDE densities:** Smooth probability curves normalized to [0,1]
  - All features on comparable scale (probability density)
  - Captures distribution shape, not raw magnitude
  - More robust to outliers and scale differences

### Why This Matters

The original DZ Grainalyzer was designed for **univariate detrital zircon age analysis**. It:
1. Treats each feature independently
2. Converts to KDE to capture distributional information
3. Factorizes the smooth density curves

This approach is specifically designed for geological data where:
- Sample sizes vary
- Measurement scales differ dramatically
- Distribution shape matters more than raw values

---

## File Changes Summary

| File | Change Type | Lines | Description |
|------|-------------|-------|-------------|
| `utils/tensor_factorization.py` | NEW FUNCTION | 38-177 | Added `create_kde_tensor_from_samples()` |
| `celery_tasks.py` | MODIFIED | 627-648 | Use KDE tensor, skip additional normalization |
| `celery_tasks.py` | MODIFIED | 671 | Remove denormalization step |
| `find_optimal_rank_modal.html` | MODIFIED | 28 | Change default max_rank from 15 to 11 |

---

## References

- Original DZ Grainalyzer: `/tmp/dzg/worker/src/helpers.jl`
  - Bandwidth calculation: lines 104
  - KDE generation: lines 105-106
  - Tensor creation: line 107
  - Rank selection: lines 129-150

- Our Implementation:
  - KDE tensor creation: `utils/tensor_factorization.py:38-177`
  - Rank selection task: `celery_tasks.py:571-787`

---

## Next Steps

After testing confirms rank 3 selection:

1. Update the other tasks (`tensor_factorization_task`, `view_empirical_kdes_task`) to also use KDE tensors
2. Verify all visualizations work correctly with KDE data
3. Update documentation to reflect KDE-based approach
4. Consider adding a toggle to switch between KDE and raw feature modes (advanced feature)
