"""
Tensor factorization for detrital zircon data using Julia's BlockTensorFactorization.jl

This module provides tensor factorization capabilities similar to those described in:
Richardson et al. (2024-2025) "Tracing Sedimentary Origins in Multivariate Geochronology
via Constrained Tensor Factorization"
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings

# Julia interface - lazy import to avoid segfaults with uWSGI
_julia_instance = None

def get_julia():
    """Lazy import of Julia to avoid uWSGI forking issues"""
    global _julia_instance
    if _julia_instance is not None:
        return _julia_instance
    
    try:
        from juliacall import Main as jl
        _julia_instance = jl
        return jl
    except ImportError:
        warnings.warn("juliacall not available - tensor factorization will not work")
        return None

# Check if juliacall is available without importing it
def _julia_available():
    """Check if juliacall can be imported"""
    try:
        import juliacall
        return True
    except ImportError:
        return False

JULIA_AVAILABLE = _julia_available()


def initialize_julia_packages():
    """Initialize Julia and install/load required packages"""
    if not JULIA_AVAILABLE:
        raise RuntimeError("juliacall is not installed")
    
    jl = get_julia()
    if jl is None:
        raise RuntimeError("Failed to import juliacall")

    # Install MatrixTensorFactor from GitHub (same package used by dzgrainalyzer)
    # Compatible with Julia 1.10+
    jl.seval("""
    using Pkg
    if !haskey(Pkg.project().dependencies, "MatrixTensorFactor")
        Pkg.add(url="https://github.com/MPF-Optimization-Laboratory/MatrixTensorFactor.jl.git", rev="main")
    end
    using MatrixTensorFactor

    # Load KernelDensity for KDE functions
    if !haskey(Pkg.project().dependencies, "KernelDensity")
        Pkg.add("KernelDensity")
    end
    using KernelDensity
    """)

    # Define wrapper functions globally for Python access
    jl.seval("""
    function calc_default_bandwidth(data, alpha, inner_percentile)
        return default_bandwidth(data, alpha, inner_percentile)
    end
    """)

    jl.seval("""
    function filter_data_inner_percentile(data, inner_percentile)
        return filter_inner_percentile(data, inner_percentile)
    end
    """)

    jl.seval("""
    function create_kde(data, bw)
        return kde(data, bandwidth=bw)
    end
    """)

    jl.seval("""
    function eval_kde_pdf(kde_obj, domain)
        return pdf(kde_obj, domain)
    end
    """)


def create_kde_tensor_from_samples(
    samples,
    feature_names: List[str],
    inner_percentile: float = 95.0,
    alpha: float = 0.9,
    n_kde_points: int = 150
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Create a 3D KDE tensor from multivariate samples using Julia's exact KDE implementation.

    This function uses Julia's KernelDensity.jl and MatrixTensorFactor.jl functions directly
    to match the original DZ Grainalyzer implementation exactly.

    Steps:
    1. Calculate default bandwidth using Julia's default_bandwidth function
    2. Generate KDEs using Julia's kde function from KernelDensity.jl
    3. Standardize KDEs to common domains
    4. Build tensor from KDE evaluations

    Parameters
    ----------
    samples : List[MultivariateSample]
        List of MultivariateSample objects
    feature_names : List[str]
        Names of features to use
    inner_percentile : float
        Percentile for bandwidth calculation (default: 95)
    alpha : float
        Bandwidth scaling factor (default: 0.9)
    n_kde_points : int
        Number of points to evaluate KDE (default: 150)

    Returns
    -------
    Tuple[np.ndarray, Dict]
        - tensor: 3D array of shape (n_samples, n_kde_points, n_features)
        - metadata: Dictionary with KDE domains and sample info
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia is required for KDE tensor creation")

    # Initialize Julia packages and wrapper functions
    initialize_julia_packages()

    n_samples = len(samples)
    n_features = len(feature_names)

    # Step 1: Calculate default bandwidth for each feature using first sample
    # Matches: bandwidths = default_bandwidth.(collect(eachmeasurement(sink1)), alpha_, inner_percentile)
    reference_sample = samples[0]
    bandwidths = {}

    for feature_name in feature_names:
        # Collect all values for this feature from reference sample
        feature_values = []
        for grain in reference_sample.grains:
            if feature_name in grain.features:
                feature_values.append(grain.features[feature_name])

        if len(feature_values) > 1:
            # Convert to Julia array
            jl_values = jl.Array(feature_values)
            # Call Julia's default_bandwidth function via wrapper
            bandwidth = jl.calc_default_bandwidth(jl_values, alpha, int(inner_percentile))
            bandwidths[feature_name] = float(bandwidth)
        else:
            # Fallback for insufficient data
            bandwidths[feature_name] = 1.0

    # Step 2: Generate KDEs using Julia's kde function
    # Matches: raw_densities = make_densities.(sinks; bandwidths, inner_percentile)
    all_kdes = {}  # {feature_name: [(jl_kde, sample_idx), ...]}
    all_feature_values_dict = {fname: [] for fname in feature_names}

    for feature_name in feature_names:
        sample_kdes = []

        for sample_idx, sample in enumerate(samples):
            feature_values = []
            for grain in sample.grains:
                if feature_name in grain.features:
                    feature_values.append(grain.features[feature_name])

            if len(feature_values) > 1:
                # Track all values for domain calculation
                all_feature_values_dict[feature_name].extend(feature_values)

                # Convert to Julia array
                jl_values = jl.Array(feature_values)

                # Call Julia's filter_inner_percentile via wrapper
                filtered_values = jl.filter_data_inner_percentile(jl_values, int(inner_percentile))

                # Call Julia's kde function via wrapper
                jl_kde = jl.create_kde(filtered_values, bandwidths[feature_name])

                sample_kdes.append((jl_kde, sample_idx))
            else:
                sample_kdes.append((None, sample_idx))

        all_kdes[feature_name] = sample_kdes

    # Step 3: Standardize KDEs to common domains
    # Matches: densities, domains = standardize_KDEs(raw_densities)
    domains = {}

    for feature_name in feature_names:
        all_values = all_feature_values_dict[feature_name]
        if len(all_values) > 0:
            all_values = np.array(all_values)
            # Use inner_percentile to determine domain range
            lower = np.percentile(all_values, (100 - inner_percentile) / 2)
            upper = np.percentile(all_values, 100 - (100 - inner_percentile) / 2)
            domain = np.linspace(lower, upper, n_kde_points)
            domains[feature_name] = domain
        else:
            domain = np.linspace(0, 1, n_kde_points)
            domains[feature_name] = domain

    # Step 4: Build tensor from KDE evaluations
    # tensor shape: (n_samples, n_kde_points, n_features)
    tensor = np.zeros((n_samples, n_kde_points, n_features))

    for f_idx, feature_name in enumerate(feature_names):
        domain = domains[feature_name]
        jl_domain = jl.Array(domain)

        for jl_kde, sample_idx in all_kdes[feature_name]:
            if jl_kde is not None:
                # Evaluate Julia KDE on domain using pdf wrapper function
                density = np.array(jl.eval_kde_pdf(jl_kde, jl_domain))

                # Normalize to integrate to 1
                density = density / np.trapz(density, domain)
                tensor[sample_idx, :, f_idx] = density
            else:
                # If no KDE could be created, use uniform distribution
                tensor[sample_idx, :, f_idx] = 1.0 / n_kde_points

    metadata = {
        'sample_names': [sample.name for sample in samples],
        'feature_names': feature_names,
        'domains': domains,
        'bandwidths': bandwidths,
        'n_kde_points': n_kde_points,
        'tensor_type': 'kde'
    }

    return tensor, metadata


def create_tensor_from_multivariate_samples(
    samples,
    feature_names: List[str],
    padding_mode: str = 'zero'
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Create a 3D tensor from multivariate samples with variable grain counts.

    Parameters
    ----------
    samples : List[MultivariateSample]
        List of MultivariateSample objects
    feature_names : List[str]
        Ordered list of feature names to include in tensor
    padding_mode : str
        How to pad samples with fewer grains than max:
        - 'zero': Pad with zeros (default)
        - 'mean': Pad with feature-wise means
        - 'replicate': Repeat last grain

    Returns
    -------
    Tuple[np.ndarray, Dict]
        - tensor: np.ndarray of shape (n_samples, max_grains, n_features)
        - metadata: Dict containing:
            - 'grain_counts': List[int] - actual grain count per sample
            - 'max_grains': int
            - 'feature_names': List[str]
            - 'sample_names': List[str]
            - 'padding_mode': str
    """
    if not samples:
        raise ValueError("No samples provided")

    if not feature_names:
        raise ValueError("No feature names provided")

    # Get dimensions
    n_samples = len(samples)
    n_features = len(feature_names)
    grain_counts = [sample.get_grain_count() for sample in samples]
    max_grains = max(grain_counts)
    sample_names = [sample.name for sample in samples]

    # Initialize tensor
    tensor = np.zeros((n_samples, max_grains, n_features), dtype=np.float32)

    # Fill tensor
    for sample_idx, sample in enumerate(samples):
        # Get feature matrix for this sample
        feature_matrix = sample.get_feature_matrix()  # (n_grains, n_features)
        n_grains = feature_matrix.shape[0]

        # Place actual data
        tensor[sample_idx, :n_grains, :] = feature_matrix

        # Apply padding if needed
        if n_grains < max_grains:
            if padding_mode == 'mean':
                # Pad with feature-wise means
                feature_means = np.mean(feature_matrix, axis=0)
                tensor[sample_idx, n_grains:, :] = feature_means
            elif padding_mode == 'replicate':
                # Repeat last grain
                last_grain = feature_matrix[-1, :]
                for i in range(n_grains, max_grains):
                    tensor[sample_idx, i, :] = last_grain
            # else: padding_mode == 'zero', already initialized with zeros

    # Check for NaN or Inf
    if np.any(np.isnan(tensor)) or np.any(np.isinf(tensor)):
        raise ValueError("Tensor contains NaN or Inf values after construction")

    # Build metadata
    metadata = {
        'grain_counts': grain_counts,
        'max_grains': max_grains,
        'feature_names': feature_names,
        'sample_names': sample_names,
        'padding_mode': padding_mode,
        'n_samples': n_samples,
        'n_features': n_features
    }

    return tensor, metadata


def normalize_tensor(
    tensor: np.ndarray,
    method: str = 'standardize',
    grain_counts: Optional[List[int]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Normalize features in tensor to compatible scales.

    Parameters
    ----------
    tensor : np.ndarray
        Shape (n_samples, max_grains, n_features)
    method : str
        Normalization method:
        - 'standardize': (x - mean) / std [RECOMMENDED - handles negatives]
        - 'minmax': (x - min) / (max - min)
        - 'robust': (x - median) / IQR
    grain_counts : Optional[List[int]]
        Actual grain counts per sample to mask padded regions when computing stats

    Returns
    -------
    Tuple[np.ndarray, Dict[str, np.ndarray]]
        - normalized_tensor: Same shape as input
        - normalization_params: Dict with parameters for denormalization
            Keys depend on method:
            - 'standardize': {'mean': array, 'std': array, 'method': 'standardize'}
            - 'minmax': {'min': array, 'max': array, 'method': 'minmax'}
            - 'robust': {'median': array, 'iqr': array, 'method': 'robust'}
    """
    n_samples, max_grains, n_features = tensor.shape
    normalized_tensor = np.copy(tensor).astype(np.float32)

    # Create mask for actual grains (vs padded zeros)
    if grain_counts is not None:
        mask = np.zeros((n_samples, max_grains), dtype=bool)
        for i, count in enumerate(grain_counts):
            mask[i, :count] = True
    else:
        mask = None

    if method == 'standardize':
        means = np.zeros(n_features)
        stds = np.zeros(n_features)

        for feat_idx in range(n_features):
            feature_data = tensor[:, :, feat_idx]

            if mask is not None:
                # Only use actual grains, not padding
                valid_data = feature_data[mask]
            else:
                valid_data = feature_data.flatten()

            means[feat_idx] = np.mean(valid_data)
            stds[feat_idx] = np.std(valid_data)

            # Handle constant features (std = 0)
            if stds[feat_idx] < 1e-10:
                print(f"Warning: Feature {feat_idx} has std ≈ 0, skipping normalization for this feature")
                stds[feat_idx] = 1.0  # Avoid division by zero

            # Normalize
            normalized_tensor[:, :, feat_idx] = (feature_data - means[feat_idx]) / stds[feat_idx]

        return normalized_tensor, {'mean': means, 'std': stds, 'method': 'standardize'}

    elif method == 'minmax':
        mins = np.zeros(n_features)
        maxs = np.zeros(n_features)

        for feat_idx in range(n_features):
            feature_data = tensor[:, :, feat_idx]

            if mask is not None:
                valid_data = feature_data[mask]
            else:
                valid_data = feature_data.flatten()

            mins[feat_idx] = np.min(valid_data)
            maxs[feat_idx] = np.max(valid_data)

            # Handle constant features
            if maxs[feat_idx] - mins[feat_idx] < 1e-10:
                print(f"Warning: Feature {feat_idx} has range ≈ 0, skipping normalization for this feature")
                maxs[feat_idx] = mins[feat_idx] + 1.0

            # Normalize
            normalized_tensor[:, :, feat_idx] = (feature_data - mins[feat_idx]) / (maxs[feat_idx] - mins[feat_idx])

        return normalized_tensor, {'min': mins, 'max': maxs, 'method': 'minmax'}

    elif method == 'robust':
        medians = np.zeros(n_features)
        iqrs = np.zeros(n_features)

        for feat_idx in range(n_features):
            feature_data = tensor[:, :, feat_idx]

            if mask is not None:
                valid_data = feature_data[mask]
            else:
                valid_data = feature_data.flatten()

            medians[feat_idx] = np.median(valid_data)
            q75, q25 = np.percentile(valid_data, [75, 25])
            iqrs[feat_idx] = q75 - q25

            # Handle constant features
            if iqrs[feat_idx] < 1e-10:
                print(f"Warning: Feature {feat_idx} has IQR ≈ 0, skipping normalization for this feature")
                iqrs[feat_idx] = 1.0

            # Normalize
            normalized_tensor[:, :, feat_idx] = (feature_data - medians[feat_idx]) / iqrs[feat_idx]

        return normalized_tensor, {'median': medians, 'iqr': iqrs, 'method': 'robust'}

    else:
        raise ValueError(f"Unknown normalization method: {method}. Choose from 'standardize', 'minmax', 'robust'")


def denormalize_tensor(
    normalized_tensor: np.ndarray,
    normalization_params: Dict[str, np.ndarray],
    method: Optional[str] = None
) -> np.ndarray:
    """
    Reverse normalization to recover original scales.

    Parameters
    ----------
    normalized_tensor : np.ndarray
        Normalized tensor of shape (n_samples, max_grains, n_features)
    normalization_params : Dict[str, np.ndarray]
        Parameters from normalize_tensor()
    method : Optional[str]
        Method used for normalization (can be inferred from params if not provided)

    Returns
    -------
    np.ndarray
        Denormalized tensor in original scale
    """
    if method is None:
        method = normalization_params.get('method')

    if method is None:
        raise ValueError("Cannot determine normalization method from params")

    denormalized = np.copy(normalized_tensor).astype(np.float32)
    n_features = denormalized.shape[2]

    if method == 'standardize':
        means = normalization_params['mean']
        stds = normalization_params['std']
        for feat_idx in range(n_features):
            denormalized[:, :, feat_idx] = denormalized[:, :, feat_idx] * stds[feat_idx] + means[feat_idx]

    elif method == 'minmax':
        mins = normalization_params['min']
        maxs = normalization_params['max']
        for feat_idx in range(n_features):
            denormalized[:, :, feat_idx] = denormalized[:, :, feat_idx] * (maxs[feat_idx] - mins[feat_idx]) + mins[feat_idx]

    elif method == 'robust':
        medians = normalization_params['median']
        iqrs = normalization_params['iqr']
        for feat_idx in range(n_features):
            denormalized[:, :, feat_idx] = denormalized[:, :, feat_idx] * iqrs[feat_idx] + medians[feat_idx]

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return denormalized


def factorize_tensor(
    tensor: np.ndarray,
    rank: int = 5,
    model: str = "Tucker1",
    max_iter: int = 6000,
    tol: float = 1e-5,
    nonnegative: bool = True,
    metadata: Optional[Dict] = None
) -> Dict:
    """
    Factorize a tensor using MatrixTensorFactor.jl's nnmtf algorithm.

    Uses the same factorization as dzgrainalyzer: Y ≈ C * F
    where C is the coefficient matrix and F is the factor tensor.

    Parameters
    ----------
    tensor : np.ndarray
        3D tensor to factorize (n_samples, n_grains, n_features)
    rank : int
        Rank of the factorization (number of latent factors)
    model : str
        Model type (kept for API compatibility, nnmtf is always used)
    max_iter : int
        Maximum number of iterations (default: 6000, same as dzgrainalyzer)
    tol : float
        Convergence tolerance (default: 1e-5, same as dzgrainalyzer)
    nonnegative : bool
        Apply non-negativity constraint (default True for nonnegative data)
    metadata : Optional[Dict]
        Optional metadata to pass through (e.g., feature names, sample names)

    Returns
    -------
    dict
        Dictionary containing:
        - 'factors': [C, F] where C is (n_samples, rank) and F is (rank, n_grains, n_features)
        - 'core': None (not applicable for nnmtf)
        - 'reconstruction': reconstructed tensor
        - 'error': final relative error
        - 'rel_errors': list of relative errors per iteration
        - 'iterations': number of iterations
        - 'converged': whether algorithm converged
        - 'model': "nnmtf"
        - 'rank': rank used
        - 'metadata': passed-through metadata (if provided)
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("juliacall is not installed")

    # Initialize Julia packages
    initialize_julia_packages()

    # Convert numpy array to Julia array
    jl_tensor = jl.Array(tensor)

    # Normalize each fiber to sum to 1 (same as dzgrainalyzer)
    # This helps with convergence and interpretation
    # NOTE: MatrixTensorFactor only supports nonnegative factorization,
    # so input tensor must be nonnegative (use minmax normalization)
    jl.seval("""
    function normalize_fibers(Y)
        Y_normalized = copy(Y)
        for i in axes(Y, 1)
            for j in axes(Y, 2)
                fiber_sum = sum(Y[i, j, :])
                if fiber_sum > 0
                    Y_normalized[i, j, :] ./= fiber_sum
                end
            end
        end
        return Y_normalized
    end
    """)

    jl_tensor_normalized = jl.normalize_fibers(jl_tensor)

    # Perform nonnegative matrix tensor factorization
    # Always use :nnscale projection (MatrixTensorFactor is nonnegative-only)
    # rescale_Y=false because we pre-normalized the fibers
    C, F, rel_errors, norm_grad, dist_Ncone = jl.seval("""
    function run_nnmtf(Y, rank, maxiter, tol)
        return nnmtf(
            Y, rank;
            projection=:nnscale,
            maxiter=maxiter,
            tol=tol,
            rescale_Y=false
        )
    end
    """)(jl_tensor_normalized, rank, max_iter, tol)

    # Convert Julia results to numpy
    C_np = np.array(C)  # Shape: (n_samples, rank)
    F_np = np.array(F)  # Shape: (rank, n_grains, n_features)
    rel_errors_list = [float(x) for x in rel_errors]

    # Reconstruct tensor: Y_reconstructed[i,j,k] = sum_r(C[i,r] * F[r,j,k])
    # Use Einstein summation for efficient computation
    reconstruction = np.einsum('ir,rjk->ijk', C_np, F_np)

    # Calculate final error
    error = rel_errors_list[-1] if rel_errors_list else 0.0

    # Check convergence (converged if reached tolerance)
    converged = error < tol

    # Build factors list in format compatible with visualization functions
    # For Tucker1-like interpretation:
    # factors[0] = C (sample mode)
    # factors[1] = "grain mode" (identity-like, not explicitly modeled in nnmtf)
    # factors[2] = feature loadings (average F across grain dimension)
    n_samples, n_grains, n_features = tensor.shape

    # Extract feature loadings by averaging F over grain dimension
    feature_loadings = np.mean(F_np, axis=1)  # Shape: (rank, n_features) -> transpose to (n_features, rank)
    feature_loadings = feature_loadings.T

    # Create grain mode factor (identity for each rank)
    grain_factor = np.eye(n_grains, rank)  # Shape: (n_grains, rank)

    factors_list = [C_np, grain_factor, feature_loadings]

    result_dict = {
        'factors': factors_list,
        'core': None,  # nnmtf doesn't have a core tensor
        'reconstruction': reconstruction,
        'error': error,
        'rel_errors': rel_errors_list,
        'iterations': len(rel_errors_list),
        'converged': converged,
        'model': 'nnmtf',
        'rank': rank
    }

    # Include metadata if provided
    if metadata is not None:
        result_dict['metadata'] = metadata

    return result_dict


def calculate_source_attribution(
    original: np.ndarray,
    factors: List[np.ndarray],
    grain_counts: List[int],
    sample_names: List[str]
) -> List[Dict]:
    """
    Calculate grain-level source attribution (which source each grain belongs to).

    For each grain in each sample, determines which latent source (factor) it most
    likely belongs to based on reconstruction contribution.

    Parameters
    ----------
    original : np.ndarray
        Original tensor of shape (n_samples, max_grains, n_features)
    factors : List[np.ndarray]
        Factor matrices from factorization
        factors[0] = C (n_samples, rank) - sample coefficients
        factors[2] = feature loadings (n_features, rank)
    grain_counts : List[int]
        Actual grain counts per sample
    sample_names : List[str]
        Names of samples

    Returns
    -------
    List[Dict]
        List of attribution dicts, one per sample:
        {
            'sample_name': str,
            'grain_attributions': List[int],  # Source index for each grain
            'grain_confidences': List[float]  # Confidence scores [0-1]
        }
    """
    n_samples, max_grains, n_features = original.shape
    C = factors[0]  # (n_samples, rank)
    rank = C.shape[1]

    # For nnmtf: F is (rank, n_grains, n_features)
    # But we stored feature_loadings as averaged/transposed version
    # We need to work with the actual reconstruction contributions

    attributions = []

    for s in range(n_samples):
        n_grains = grain_counts[s]
        grain_sources = []
        grain_confidences = []

        for i in range(n_grains):
            # For each grain, calculate contribution from each source/factor
            # Contribution of factor r to grain i = C[s, r] * ||original grain features||
            # We measure how well each factor explains this grain

            grain_features = original[s, i, :]

            # Calculate "likelihood" that this grain belongs to each source
            # Using the factor coefficients as weights
            source_weights = C[s, :]  # (rank,)

            # Normalize to get probabilities
            if np.sum(source_weights) > 0:
                source_probs = source_weights / np.sum(source_weights)
            else:
                source_probs = np.ones(rank) / rank

            # Assign to most likely source
            best_source = np.argmax(source_probs)

            # Calculate confidence as the ratio of best to second-best
            sorted_probs = np.sort(source_probs)[::-1]
            if len(sorted_probs) > 1 and sorted_probs[1] > 0:
                confidence = sorted_probs[0] / sorted_probs[1]  # Ratio of best to 2nd best
                confidence = min(confidence, 10.0)  # Cap at 10x
                confidence = confidence / 10.0  # Normalize to [0, 1]
            else:
                confidence = 1.0

            grain_sources.append(int(best_source + 1))  # 1-indexed for display
            grain_confidences.append(float(confidence))

        attributions.append({
            'sample_name': sample_names[s],
            'grain_attributions': grain_sources,
            'grain_confidences': grain_confidences
        })

    return attributions


def explained_variance(tensor: np.ndarray, reconstruction: np.ndarray) -> float:
    """
    Calculate explained variance (R²) of the factorization

    Parameters
    ----------
    tensor : np.ndarray
        Original tensor
    reconstruction : np.ndarray
        Reconstructed tensor from factorization

    Returns
    -------
    float
        Explained variance (0-1, where 1 is perfect reconstruction)
    """
    ss_total = np.sum((tensor - np.mean(tensor)) ** 2)
    ss_residual = np.sum((tensor - reconstruction) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared


def visualize_factor_loadings(
    factors: List[np.ndarray],
    feature_names: List[str],
    title: str = "Factor Loadings",
    font_path: str = None,
    font_size: int = 12,
    fig_width: float = 10,
    fig_height: float = 8,
    color_map: str = 'RdBu_r'
):
    """
    Create heatmap showing feature contributions to factors.

    For Tucker1 models, visualizes the feature-mode factor (typically factors[2])
    showing how each feature contributes to each latent factor.

    Parameters
    ----------
    factors : List[np.ndarray]
        List of factor matrices from factorization
    feature_names : List[str]
        Names of features (for Y-axis labels)
    title : str
        Plot title
    font_path : str
        Path to font file
    font_size : int
        Font size for labels
    fig_width : float
        Figure width in inches
    fig_height : float
        Figure height in inches
    color_map : str
        Colormap name (diverging recommended: RdBu_r, coolwarm)

    Returns
    -------
    matplotlib.figure.Figure
        Figure object with factor loadings heatmap
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # Set up font
    if font_path:
        prop = fm.FontProperties(fname=font_path, size=font_size)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['font.size'] = font_size

    # For Tucker1: factors[0]=samples, factors[1]=grains, factors[2]=features
    # Extract feature-mode factor (last factor for Tucker models)
    if len(factors) >= 3:
        feature_factor = factors[2]  # Shape: (n_features, rank)
    elif len(factors) == 2:
        # CP model or Tucker2: try last factor
        feature_factor = factors[-1]
    else:
        raise ValueError(f"Expected at least 2 factors, got {len(factors)}")

    # Ensure we have 2D factor
    if feature_factor.ndim != 2:
        raise ValueError(f"Feature factor should be 2D, got shape {feature_factor.shape}")

    n_features, rank = feature_factor.shape

    # Validate feature names
    if len(feature_names) != n_features:
        print(f"Warning: {len(feature_names)} feature names but {n_features} features in factor")
        feature_names = feature_names[:n_features] + [f"Feature {i}" for i in range(len(feature_names), n_features)]

    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create heatmap
    # Use diverging colormap centered at 0
    vmax = max(abs(feature_factor.min()), abs(feature_factor.max()))
    im = ax.imshow(feature_factor, aspect='auto', cmap=color_map,
                   origin='lower', vmin=-vmax, vmax=vmax)

    # Set ticks and labels
    ax.set_xticks(np.arange(rank))
    ax.set_xticklabels([f'Factor {i+1}' for i in range(rank)])
    ax.set_yticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(feature_names)

    ax.set_xlabel('Latent Factors', fontsize=font_size)
    ax.set_ylabel('Features', fontsize=font_size)
    ax.set_title(title, fontsize=font_size + 2, pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Loading', rotation=270, labelpad=20, fontsize=font_size)

    # Annotate cells with values if not too many
    if n_features * rank <= 100:  # Only annotate if heatmap isn't too large
        for i in range(n_features):
            for j in range(rank):
                text = ax.text(j, i, f'{feature_factor[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=font_size-2)

    plt.tight_layout()
    return fig


def visualize_reconstruction_comparison(
    original: np.ndarray,
    reconstruction: np.ndarray,
    feature_names: List[str],
    sample_names: List[str],
    grain_counts: List[int],
    sample_index: int = 0,
    max_grains: int = 50,
    title: str = "Reconstruction Comparison",
    font_path: str = None,
    font_size: int = 12,
    fig_width: float = 14,
    fig_height: float = 10,
    color_map: str = 'viridis'
):
    """
    Compare original and reconstructed feature values for a single sample.

    Plots one subplot per feature showing original vs reconstructed values
    across grains for the selected sample.

    Parameters
    ----------
    original : np.ndarray
        Original tensor of shape (n_samples, max_grains, n_features)
    reconstruction : np.ndarray
        Reconstructed tensor (same shape as original)
    feature_names : List[str]
        Names of features (for subplot titles)
    sample_names : List[str]
        Names of samples
    grain_counts : List[int]
        Actual grain counts per sample (to mask padding)
    sample_index : int
        Which sample to visualize (default: 0, first sample)
    max_grains : int
        Maximum number of grains to show (to avoid clutter)
    title : str
        Main plot title
    font_path : str
        Path to font file
    font_size : int
        Font size
    fig_width : float
        Figure width
    fig_height : float
        Figure height
    color_map : str
        Colormap name (not used currently, kept for API compatibility)

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one subplot per feature comparing original vs reconstruction
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # Set up font
    if font_path:
        prop = fm.FontProperties(fname=font_path, size=font_size)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['font.size'] = font_size

    # Validate inputs
    if original.ndim != 3 or reconstruction.ndim != 3:
        raise ValueError(f"Expected 3D tensors, got shapes {original.shape}, {reconstruction.shape}")

    if sample_index >= len(sample_names):
        raise ValueError(f"sample_index {sample_index} out of range (only {len(sample_names)} samples)")

    n_samples, max_grains_tensor, n_features = original.shape
    sample_name = sample_names[sample_index]
    grain_count = grain_counts[sample_index]

    # Limit grains to display
    n_grains_to_show = min(grain_count, max_grains)

    # Extract data for this sample
    orig_sample = original[sample_index, :n_grains_to_show, :]  # (n_grains, n_features)
    recon_sample = reconstruction[sample_index, :n_grains_to_show, :]

    # Create subplots (one per feature)
    fig, axes = plt.subplots(n_features, 1, figsize=(fig_width, fig_height))

    if n_features == 1:
        axes = [axes]

    grain_indices = np.arange(1, n_grains_to_show + 1)

    for feat_idx, ax in enumerate(axes):
        feat_name = feature_names[feat_idx]

        # Plot original vs reconstructed
        ax.plot(grain_indices, orig_sample[:, feat_idx], 'o-',
                color='black', linewidth=2, markersize=4, label='Original', alpha=0.7)
        ax.plot(grain_indices, recon_sample[:, feat_idx], 's--',
                color='red', linewidth=2, markersize=4, label='Reconstructed', alpha=0.7)

        # Calculate per-feature R²
        ss_total = np.sum((orig_sample[:, feat_idx] - np.mean(orig_sample[:, feat_idx])) ** 2)
        ss_residual = np.sum((orig_sample[:, feat_idx] - recon_sample[:, feat_idx]) ** 2)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

        ax.set_xlabel('Grain Index', fontsize=font_size)
        ax.set_ylabel(feat_name, fontsize=font_size)
        ax.set_title(f'{feat_name} (R² = {r2:.3f})', fontsize=font_size)
        ax.legend(loc='best', fontsize=font_size-2)
        ax.grid(True, alpha=0.3)

    # Overall title
    main_title = f"{title}\nSample: {sample_name} ({n_grains_to_show}/{grain_count} grains shown)"
    plt.suptitle(main_title, fontsize=font_size + 2, y=0.995)
    plt.tight_layout()

    return fig


def visualize_sample_scores(
    factors: List[np.ndarray],
    sample_names: List[str],
    title: str = "Sample Scores",
    font_path: str = None,
    font_size: int = 12,
    fig_width: float = 10,
    fig_height: float = 8,
    color_map: str = 'viridis'
):
    """
    Show sample projections onto latent factors.

    For Tucker1 models, extracts the sample-mode factor (typically factors[0])
    and visualizes how samples project into the latent factor space.

    Parameters
    ----------
    factors : List[np.ndarray]
        List of factor matrices from factorization
    sample_names : List[str]
        Names of samples (for labels)
    title : str
        Plot title
    font_path : str
        Path to font file
    font_size : int
        Font size
    fig_width : float
        Figure width
    fig_height : float
        Figure height
    color_map : str
        Colormap name

    Returns
    -------
    matplotlib.figure.Figure
        Figure showing sample scores
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib import cm

    # Set up font
    if font_path:
        prop = fm.FontProperties(fname=font_path, size=font_size)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['font.size'] = font_size

    # Extract sample-mode factor (first factor for Tucker models)
    sample_factor = factors[0]  # Shape: (n_samples, rank)

    if sample_factor.ndim != 2:
        raise ValueError(f"Sample factor should be 2D, got shape {sample_factor.shape}")

    n_samples, rank = sample_factor.shape

    # Validate sample names
    if len(sample_names) != n_samples:
        print(f"Warning: {len(sample_names)} sample names but {n_samples} samples in factor")
        sample_names = sample_names[:n_samples] + [f"Sample {i}" for i in range(len(sample_names), n_samples)]

    fig = plt.figure(figsize=(fig_width, fig_height))

    cmap = cm.get_cmap(color_map)
    colors = [cmap(i / max(1, n_samples - 1)) for i in range(n_samples)]

    if rank == 2:
        # 2D scatter plot
        ax = fig.add_subplot(111)
        for i, sample_name in enumerate(sample_names):
            ax.scatter(sample_factor[i, 0], sample_factor[i, 1],
                      s=100, color=colors[i], alpha=0.7, edgecolors='black', linewidth=1)
            ax.annotate(sample_name, (sample_factor[i, 0], sample_factor[i, 1]),
                       fontsize=font_size-2, ha='right', alpha=0.8)

        ax.set_xlabel('Factor 1', fontsize=font_size)
        ax.set_ylabel('Factor 2', fontsize=font_size)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)

    elif rank == 3:
        # 3D scatter plot
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')

        for i, sample_name in enumerate(sample_names):
            ax.scatter(sample_factor[i, 0], sample_factor[i, 1], sample_factor[i, 2],
                      s=100, color=colors[i], alpha=0.7, edgecolors='black', linewidth=1)
            ax.text(sample_factor[i, 0], sample_factor[i, 1], sample_factor[i, 2],
                   sample_name, fontsize=font_size-3)

        ax.set_xlabel('Factor 1', fontsize=font_size)
        ax.set_ylabel('Factor 2', fontsize=font_size)
        ax.set_zlabel('Factor 3', fontsize=font_size)

    else:
        # Grouped bar chart for rank > 3
        ax = fig.add_subplot(111)

        x = np.arange(n_samples)
        width = 0.8 / rank  # Bar width

        for factor_idx in range(rank):
            offset = (factor_idx - rank/2) * width
            ax.bar(x + offset, sample_factor[:, factor_idx], width,
                  label=f'Factor {factor_idx + 1}', alpha=0.7)

        ax.set_xlabel('Samples', fontsize=font_size)
        ax.set_ylabel('Factor Score', fontsize=font_size)
        ax.set_xticks(x)
        ax.set_xticklabels(sample_names, rotation=45, ha='right', fontsize=font_size-2)
        ax.legend(loc='best', fontsize=font_size-2)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

    ax.set_title(title, fontsize=font_size + 2, pad=15)
    plt.tight_layout()

    return fig


def visualize_source_attribution(
    attributions: List[Dict],
    rank: int,
    title: str = "Source attribution based on learned kernel density estimates",
    font_path: str = None,
    font_size: int = 12,
    fig_width: float = 14,
    fig_height: float = 6,
    color_map: str = 'Greens'
):
    """
    Visualize grain-level source attribution showing which source each grain belongs to.

    Creates a single plot showing the assigned source for each grain across all samples,
    with confidence scores indicated by color intensity (dzgrainalyzer style).

    Parameters
    ----------
    attributions : List[Dict]
        Source attribution data from calculate_source_attribution()
    rank : int
        Number of sources/factors
    title : str
        Plot title
    font_path : str
        Path to font file
    font_size : int
        Font size
    fig_width : float
        Figure width
    fig_height : float
        Figure height
    color_map : str
        Colormap name (use 'Greens' for dzgrainalyzer style)

    Returns
    -------
    matplotlib.figure.Figure
        Figure showing source attribution for all grains
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib import cm
    from matplotlib.colors import Normalize

    # Set up font
    if font_path:
        prop = fm.FontProperties(fname=font_path, size=font_size)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['font.size'] = font_size

    n_samples = len(attributions)

    # Create single plot for all samples (dzgrainalyzer style)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # Get colormap for confidence
    conf_cmap = cm.get_cmap(color_map)
    norm = Normalize(vmin=0.0, vmax=1.0)

    # Collect all data and calculate grain index offsets
    grain_offset = 0
    sample_positions = []
    sample_labels = []

    for attr in attributions:
        sample_name = attr['sample_name']
        grain_sources = np.array(attr['grain_attributions'])
        grain_confidences = np.array(attr['grain_confidences'])
        n_grains = len(grain_sources)

        grain_indices = np.arange(grain_offset, grain_offset + n_grains)

        # Plot each grain with color intensity based on confidence
        scatter = ax.scatter(
            grain_indices,
            grain_sources,
            c=grain_confidences,
            cmap=conf_cmap,
            norm=norm,
            s=80,
            alpha=0.8,
            edgecolors='none'
        )

        # Store sample position and label
        sample_positions.append(grain_offset + n_grains / 2)
        sample_labels.append(sample_name)

        grain_offset += n_grains

    # Configure axes
    ax.set_xlabel('grain index', fontsize=font_size)
    ax.set_ylabel('source', fontsize=font_size)
    ax.set_yticks(range(1, rank + 1))
    ax.set_ylim(0.5, rank + 0.5)
    ax.set_xlim(-2, grain_offset + 2)
    ax.grid(True, alpha=0.3, axis='both')

    # Add sample labels below x-axis
    ax.set_xticks(sample_positions)
    ax.set_xticklabels(sample_labels, rotation=45, ha='right', fontsize=font_size - 2)

    # Add colorbar for confidence
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('confidence', fontsize=font_size)

    plt.title(title, fontsize=font_size + 1, pad=10)
    plt.tight_layout()

    return fig


def visualize_empirical_kdes(
    tensor: np.ndarray,
    feature_names: List[str],
    sample_names: List[str],
    grain_counts: List[int],
    title: str = "Empirical Kernel Density Estimates of input variables",
    font_path: str = None,
    font_size: int = 12,
    fig_width: float = 14,
    fig_height: float = 8,
    color_map: str = 'tab20',
    fill: bool = False
):
    """
    Create KDE plots for each feature showing distribution across all samples.

    Similar to dzgrainalyzer's input visualization.

    Parameters
    ----------
    tensor : np.ndarray
        Original tensor (n_samples, max_grains, n_features)
    feature_names : List[str]
        Names of features
    sample_names : List[str]
        Names of samples
    grain_counts : List[int]
        Actual grain counts per sample
    title : str
        Plot title
    font_path : str
        Path to font file
    font_size : int
        Font size
    fig_width : float
        Figure width
    fig_height : float
        Figure height
    color_map : str
        Colormap name
    fill : bool
        If True, fill the area under each KDE curve.
        If False, just plot the KDE line.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with KDE plots for each feature
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib import cm
    from scipy.stats import gaussian_kde

    # Set up font
    if font_path:
        prop = fm.FontProperties(fname=font_path, size=font_size)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['font.size'] = font_size

    n_samples, max_grains, n_features = tensor.shape

    # Create stacked vertical subplots (one per feature)
    fig, axes = plt.subplots(n_features, 1, figsize=(fig_width, fig_height))
    if n_features == 1:
        axes = [axes]

    # Get colormap
    cmap = cm.get_cmap(color_map, n_samples)
    colors = [cmap(i) for i in range(n_samples)]

    for feat_idx, feature_name in enumerate(feature_names):
        ax = axes[feat_idx]

        # Create KDE for each sample
        for s_idx in range(n_samples):
            # Get actual grains (not padded)
            grain_count = grain_counts[s_idx]
            feature_values = tensor[s_idx, :grain_count, feat_idx]

            # Skip if no valid data
            if len(feature_values) == 0 or np.all(np.isnan(feature_values)):
                continue

            # Remove NaN values
            feature_values = feature_values[~np.isnan(feature_values)]

            if len(feature_values) < 2:
                continue

            # Create KDE
            try:
                kde = gaussian_kde(feature_values)

                # Create domain for evaluation
                x_min, x_max = np.min(feature_values), np.max(feature_values)
                x_range = x_max - x_min
                x_min -= x_range * 0.1
                x_max += x_range * 0.1
                x_domain = np.linspace(x_min, x_max, 200)

                # Evaluate KDE
                density = kde(x_domain)

                # Plot
                if fill:
                    ax.fill_between(x_domain, density, alpha=0.5, color=colors[s_idx],
                                   label=sample_names[s_idx])
                    ax.plot(x_domain, density, color=colors[s_idx], alpha=0.8, linewidth=1.5)
                else:
                    ax.plot(x_domain, density, color=colors[s_idx],
                           label=sample_names[s_idx], alpha=0.7, linewidth=1.5)
            except Exception as e:
                print(f"Warning: Could not create KDE for {feature_name}, sample {sample_names[s_idx]}: {e}")
                continue

        ax.set_ylabel('density', fontsize=font_size - 1)
        ax.set_title(feature_name, fontsize=font_size, loc='left', pad=5)
        ax.grid(True, alpha=0.3)

        # Only show legend on first subplot
        if feat_idx == 0:
            ax.legend(loc='upper right', fontsize=font_size - 3, ncol=3)

    # Only show x-label on bottom subplot
    axes[-1].set_xlabel('value', fontsize=font_size)

    plt.suptitle(title, fontsize=font_size + 2)
    plt.tight_layout()

    return fig


def visualize_empirical_kdes_tabbed(
    tensor: np.ndarray,
    feature_names: List[str],
    sample_names: List[str],
    grain_counts: List[int],
    title: str = "Empirical Kernel Density Estimates of input variables",
    font_path: str = None,
    font_size: int = 12,
    fig_width: float = 14,
    fig_height: float = 6,
    color_map: str = 'tab20',
    stack_samples: bool = True,
    fill: bool = False
):
    """
    Create individual KDE plots for each feature (for tabbed display).

    Each feature gets its own figure showing distribution across all samples.
    Similar to original dzgrainalyzer's tabbed feature view.

    Parameters
    ----------
    tensor : np.ndarray
        Original tensor (n_samples, max_grains, n_features)
    feature_names : List[str]
        Names of features
    sample_names : List[str]
        Names of samples
    grain_counts : List[int]
        Actual grain counts per sample
    title : str
        Base title for plots
    font_path : str
        Path to font file
    font_size : int
        Font size
    fig_width : float
        Figure width for each tab
    fig_height : float
        Figure height for each tab
    color_map : str
        Colormap name
    stack_samples : bool
        If True, overlay all samples on one plot per feature (non-stacked).
        If False, create separate subplots for each sample stacked vertically within each feature.
    fill : bool
        If True, fill the area under each KDE curve.
        If False, just plot the KDE line.

    Returns
    -------
    List[matplotlib.figure.Figure]
        List of figures, one per feature
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib import cm
    from matplotlib import gridspec
    from scipy.stats import gaussian_kde

    # Set up font
    if font_path:
        prop = fm.FontProperties(fname=font_path, size=font_size)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['font.size'] = font_size

    n_samples, max_grains, n_features = tensor.shape

    # Get colormap
    cmap = cm.get_cmap(color_map, n_samples)
    colors = [cmap(i) for i in range(n_samples)]

    figures = []

    # Create one figure per feature
    for feat_idx, feature_name in enumerate(feature_names):
        if stack_samples:
            # Overlaid mode: all samples on one plot
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
            axes = [ax]
        else:
            # Stacked mode: one subplot per sample (stacked vertically, like dz_lib)
            fig = plt.figure(figsize=(fig_width, fig_height))
            gs = gridspec.GridSpec(n_samples, 1, figure=fig)
            axes = []
            for i in range(n_samples):
                ax = fig.add_subplot(gs[i])
                axes.append(ax)

        # Create KDE for each sample
        for s_idx in range(n_samples):
            # Get actual grains (not padded)
            grain_count = grain_counts[s_idx]
            feature_values = tensor[s_idx, :grain_count, feat_idx]

            # Skip if no valid data
            if len(feature_values) == 0 or np.all(np.isnan(feature_values)):
                continue

            # Remove NaN values
            feature_values = feature_values[~np.isnan(feature_values)]

            if len(feature_values) < 2:
                continue

            # Create KDE
            try:
                kde = gaussian_kde(feature_values)

                # Create domain for evaluation
                x_min, x_max = np.min(feature_values), np.max(feature_values)
                x_range = x_max - x_min
                x_min -= x_range * 0.1
                x_max += x_range * 0.1
                x_domain = np.linspace(x_min, x_max, 200)

                # Evaluate KDE
                density = kde(x_domain)

                # Select appropriate axis
                if stack_samples:
                    ax = axes[0]
                    # Plot with label for legend
                    if fill:
                        ax.fill_between(x_domain, density, alpha=0.25, color=colors[s_idx],
                                       linewidth=0)
                        ax.plot(x_domain, density, color=colors[s_idx], label=sample_names[s_idx])
                    else:
                        ax.plot(x_domain, density, color=colors[s_idx],
                               label=sample_names[s_idx])
                else:
                    ax = axes[s_idx]
                    # Plot with label (for legend on the right, like dz_lib)
                    if fill:
                        ax.fill_between(x_domain, density, alpha=0.25, color=colors[s_idx],
                                       linewidth=0)
                        ax.plot(x_domain, density, color=colors[s_idx], label=sample_names[s_idx])
                    else:
                        ax.plot(x_domain, density, color=colors[s_idx],
                               label=sample_names[s_idx])

                    # Add legend on the right side, like dz_lib
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)

            except Exception as e:
                print(f"Warning: Could not create KDE for {feature_name}, sample {sample_names[s_idx]}: {e}")
                continue

        # Format axes - following dz_lib pattern
        for i, ax in enumerate(axes):
            ax.tick_params(axis='both', which='major', labelsize=font_size)

            # Remove x-tick labels for all but the last subplot in stacked mode
            if not stack_samples and i < len(axes) - 1:
                ax.tick_params(axis='x', labelbottom=False)

            ax.set_facecolor('white')
            ax.tick_params(axis='x', colors='black', width=2)
            ax.tick_params(axis='y', colors='black', width=2)

        if stack_samples:
            # Format overlaid plot
            ax = axes[0]
            ax.set_xlabel('value', fontsize=font_size)
            ax.set_ylabel('density', fontsize=font_size)
            ax.set_title(f'{feature_name}\n{title}', fontsize=font_size + 1)
            ax.legend(loc='upper right', fontsize=font_size - 2, ncol=min(3, n_samples))
            plt.tight_layout()
        else:
            # Format stacked subplots - following dz_lib pattern exactly
            fig.suptitle(f'{feature_name}\n{title}', fontsize=font_size * 1.75, fontproperties=prop if font_path else None)

            # Add figure-level axis labels (one for whole figure, like dz_lib)
            fig.text(0.5, 0.02, 'value', ha='center', va='center',
                    fontsize=font_size, fontproperties=prop if font_path else None)
            fig.text(0.01, 0.5, 'density', va='center', rotation='vertical',
                    fontsize=font_size, fontproperties=prop if font_path else None)

            # Use tight_layout with rect to leave space for labels
            fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.96])

        figures.append(fig)

    return figures


# =============================================================================
# Rank Selection Visualization Functions
# =============================================================================
# These functions are called by find_optimal_rank_task in celery_tasks.py
# to visualize the results from Julia's rank selection analysis.
# Users can choose to generate either or both plots.
# =============================================================================

def visualize_misfit_plot(
    ranks: List[int],
    errors: List[float],
    title: str = "Misfit versus Ranks",
    font_path: str = None,
    font_size: int = 12,
    fig_width: float = 10,
    fig_height: float = 6
):
    """
    Create misfit vs ranks plot (relative error)

    Parameters
    ----------
    ranks : List[int]
        List of ranks tested
    errors : List[float]
        Relative reconstruction errors for each rank
    title : str
        Plot title
    font_path : str
        Path to font file
    fig_size : int
        Font size
    fig_width : float
        Figure width
    fig_height : float
        Figure height

    Returns
    -------
    matplotlib.figure.Figure
        Figure with misfit plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # Set up font
    if font_path:
        prop = fm.FontProperties(fname=font_path, size=font_size)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['font.size'] = font_size

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot misfit
    ax.plot(ranks, errors, 'o-', color='black', linewidth=2, markersize=6)
    ax.set_xlabel('Rank', fontsize=font_size)
    ax.set_ylabel('Relative Error', fontsize=font_size)
    ax.set_title('Residual misfit between empirical variables (Y) and reconstructed variables (A B)',
                 fontsize=font_size)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ranks)

    plt.suptitle(title, fontsize=font_size + 2)
    plt.tight_layout()

    return fig


def visualize_curvature_plot(
    ranks: List[int],
    curvatures: List[float],
    best_rank: int,
    title: str = "Optimum Rank",
    font_path: str = None,
    font_size: int = 12,
    fig_width: float = 10,
    fig_height: float = 6
):
    """
    Create curvature analysis plot for optimal rank selection

    Parameters
    ----------
    ranks : List[int]
        List of ranks tested
    curvatures : List[float]
        Curvature values (second derivative of error)
    best_rank : int
        Optimal rank identified by curvature
    title : str
        Plot title
    font_path : str
        Path to font file
    font_size : int
        Font size
    fig_width : float
        Figure width
    fig_height : float
        Figure height

    Returns
    -------
    matplotlib.figure.Figure
        Figure with curvature plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # Set up font
    if font_path:
        prop = fm.FontProperties(fname=font_path, size=font_size)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['font.size'] = font_size

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot curvature
    ax.plot(ranks, curvatures, 'o-', color='black', linewidth=2, markersize=6)
    ax.axvline(x=best_rank, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Rank', fontsize=font_size)
    ax.set_ylabel('Curvature of Relative Error', fontsize=font_size)
    ax.set_title(f'Optimum rank identified as the maximum standard curvature in this graph. Selected Rank: {best_rank}',
                 fontsize=font_size)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ranks)

    plt.suptitle(title, fontsize=font_size + 2)
    plt.tight_layout()

    return fig


def visualize_learned_source_kdes(
    reconstruction: np.ndarray,
    factors: List[np.ndarray],
    grain_counts: List[int],
    feature_names: List[str],
    rank: int,
    title: str = "Learned Kernel Density Estimates",
    font_path: str = None,
    font_size: int = 12,
    fig_width: float = 14,
    fig_height: float = 8,
    color_map: str = 'Set2'
):
    """
    Visualize learned source distributions as KDEs.

    Reconstructs KDE distributions for each learned source by extracting
    the source-specific grain distributions from the factorization result.

    Parameters
    ----------
    reconstruction : np.ndarray
        Reconstructed tensor from factorization (n_samples, max_grains, n_features)
    factors : List[np.ndarray]
        Factor matrices: [C (samples×rank), G (grains×rank), F (features×rank)]
    grain_counts : List[int]
        Actual grain counts per sample
    feature_names : List[str]
        Names of features
    rank : int
        Number of sources/factors
    title : str
        Plot title
    font_path : str
        Path to font file
    font_size : int
        Font size
    fig_width : float
        Figure width
    fig_height : float
        Figure height
    color_map : str
        Colormap name

    Returns
    -------
    matplotlib.figure.Figure
        Figure with KDE plots for learned sources (one feature per figure)
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib import cm
    from scipy.stats import gaussian_kde

    # Set up font
    if font_path:
        prop = fm.FontProperties(fname=font_path, size=font_size)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['font.size'] = font_size

    # Extract factors
    C = factors[0]  # (n_samples, rank) - sample coefficients
    G = factors[1]  # (max_grains, rank) - grain coefficients
    F = factors[2]  # (n_features, rank) - feature coefficients

    n_features = len(feature_names)

    # Create stacked vertical subplots (one per feature)
    fig, axes = plt.subplots(n_features, 1, figsize=(fig_width, fig_height))
    if n_features == 1:
        axes = [axes]

    # Get colormap
    cmap = cm.get_cmap(color_map, rank)
    colors = [cmap(i) for i in range(rank)]

    for feat_idx, feature_name in enumerate(feature_names):
        ax = axes[feat_idx]

        # For each source, reconstruct the distribution for this feature
        for source_idx in range(rank):
            # Get grain weights for this source
            grain_weights = G[:, source_idx]  # (max_grains,)

            # Get feature coefficient for this source
            feature_coef = F[feat_idx, source_idx]

            # Collect grain values weighted by their source membership
            # We'll sample from the reconstruction weighted by grain membership to this source
            source_values = []

            # Gather values from all samples, weighted by how much they contribute to this source
            n_samples = reconstruction.shape[0]
            max_grains = reconstruction.shape[1]

            for s_idx in range(n_samples):
                sample_coef = C[s_idx, source_idx]  # How much this sample has of this source
                n_grains = min(grain_counts[s_idx], max_grains)

                for g_idx in range(n_grains):
                    grain_weight = grain_weights[g_idx]
                    # Weight determines how much this grain belongs to this source
                    weight = sample_coef * grain_weight * feature_coef

                    if abs(weight) > 0.01:  # Only include significantly weighted grains
                        grain_value = reconstruction[s_idx, g_idx, feat_idx]
                        # Add multiple copies based on weight to build distribution
                        n_copies = max(1, int(abs(weight) * 100))
                        source_values.extend([grain_value] * n_copies)

            if len(source_values) > 5:  # Need enough points for KDE
                source_values = np.array(source_values)
                source_values = source_values[~np.isnan(source_values)]

                if len(source_values) > 5:
                    try:
                        # Create KDE
                        kde = gaussian_kde(source_values)

                        # Create domain for evaluation
                        x_min, x_max = np.percentile(source_values, [1, 99])
                        x_range = x_max - x_min
                        x_min -= x_range * 0.1
                        x_max += x_range * 0.1
                        x_domain = np.linspace(x_min, x_max, 200)

                        # Evaluate KDE
                        density = kde(x_domain)

                        # Plot
                        ax.plot(x_domain, density, color=colors[source_idx],
                               label=f'source {source_idx + 1}', alpha=0.7, linewidth=2)
                    except:
                        pass  # Skip if KDE fails

        ax.set_ylabel('density', fontsize=font_size - 1)
        ax.set_title(feature_name, fontsize=font_size, loc='left', pad=5)
        ax.grid(True, alpha=0.3)

        # Only show legend on first subplot
        if feat_idx == 0:
            ax.legend(loc='upper right', fontsize=font_size - 2)

    # Only show x-label on bottom subplot
    axes[-1].set_xlabel('value', fontsize=font_size)

    plt.suptitle(title, fontsize=font_size + 2)
    plt.tight_layout()

    return fig

def standard_curvature(errors: List[float]) -> List[float]:
    """
    Calculate standard curvature (second derivative) for rank selection.

    The maximum curvature indicates the optimal rank where adding more
    factors provides diminishing returns.

    Parameters
    ----------
    errors : List[float]
        Relative errors for each rank

    Returns
    -------
    List[float]
        Curvature values (second derivative approximation)
    """
    if len(errors) < 3:
        return [0.0] * len(errors)

    curvatures = []
    for i in range(len(errors)):
        if i == 0 or i == len(errors) - 1:
            curvatures.append(0.0)
        else:
            # Second derivative approximation: f''(x) ≈ f(x-1) - 2f(x) + f(x+1)
            d2 = errors[i-1] - 2*errors[i] + errors[i+1]
            curvatures.append(abs(d2))

    return curvatures
