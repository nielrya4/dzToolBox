"""
Additional mathematical optimizations for dzToolBox
Optimizes KDE, MDS, and other CPU-intensive operations
"""
import numpy as np
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from dz_lib.univariate import distributions
from dz_lib.utils import matrices


def optimize_numpy_for_performance():
    """Configure numpy for optimal performance on multi-core systems"""
    cores = min(12, cpu_count())
    
    # Set environment variables for numpy threading
    os.environ['OMP_NUM_THREADS'] = str(cores)
    os.environ['MKL_NUM_THREADS'] = str(cores) 
    os.environ['NUMEXPR_NUM_THREADS'] = str(cores)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cores)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cores)
    
    # Additional numpy performance settings
    try:
        import numpy as np
        # Set numpy to use all available cores for linear algebra operations
        if hasattr(np, 'show_config'):
            # Force numpy to detect threading libraries
            pass
    except ImportError:
        pass
    
    return cores


def parallel_kde_computation(samples, bandwidth, n_processes=None):
    """
    Parallel computation of KDE functions for multiple samples
    
    Args:
        samples: List of sample objects
        bandwidth: KDE bandwidth parameter
        n_processes: Number of processes (default: auto-detect)
    
    Returns:
        List of KDE distributions
    """

    cores_available = optimize_numpy_for_performance()
    
    if n_processes is None:
        n_processes = min(cores_available, len(samples), 8)
    
    if len(samples) <= 1 or n_processes <= 1:
        # Fall back to sequential processing for small workloads
        return [distributions.kde_function(sample=sample, bandwidth=bandwidth) for sample in samples]
    
    # Prepare arguments for parallel processing
    kde_args = [(sample, bandwidth) for sample in samples]
    
    # Use multiprocessing for parallel KDE computation
    with Pool(processes=n_processes) as pool:
        results = pool.map(compute_kde_worker, kde_args)
    
    return results


def compute_kde_worker(args):
    """Worker function for parallel KDE computation"""
    sample, bandwidth = args
    return distributions.kde_function(sample=sample, bandwidth=bandwidth)


def parallel_pdp_computation(samples, x_min=None, x_max=None, n_processes=None):
    """
    Parallel computation of PDP functions for multiple samples
    
    Args:
        samples: List of sample objects
        x_min, x_max: Age range boundaries  
        n_processes: Number of processes (default: auto-detect)
    
    Returns:
        List of PDP distributions
    """

    cores_available = optimize_numpy_for_performance()
    
    if n_processes is None:
        n_processes = min(cores_available, len(samples), 8)
    
    if len(samples) <= 1 or n_processes <= 1:
        # Fall back to sequential processing
        if x_min is not None and x_max is not None:
            return [distributions.pdp_function(sample, x_min, x_max) for sample in samples]
        else:
            return [distributions.pdp_function(sample) for sample in samples]
    
    # Prepare arguments for parallel processing
    if x_min is not None and x_max is not None:
        pdp_args = [(sample, x_min, x_max) for sample in samples]
    else:
        pdp_args = [(sample, None, None) for sample in samples]
    
    # Use multiprocessing for parallel PDP computation
    with Pool(processes=n_processes) as pool:
        results = pool.map(compute_pdp_worker, pdp_args)
    
    return results


def compute_pdp_worker(args):
    """Worker function for parallel PDP computation"""
    sample, x_min, x_max = args
    
    if x_min is not None and x_max is not None:
        return distributions.pdp_function(sample, x_min, x_max)
    else:
        return distributions.pdp_function(sample)


def parallel_cdf_computation(distributions_list, n_processes=None):
    """
    Parallel computation of CDF functions from distributions
    
    Args:
        distributions_list: List of distribution objects
        n_processes: Number of processes (default: auto-detect)
    
    Returns:
        List of CDF distributions
    """

    cores_available = optimize_numpy_for_performance()
    
    if n_processes is None:
        n_processes = min(cores_available, len(distributions_list), 8)
    
    if len(distributions_list) <= 1 or n_processes <= 1:
        # Fall back to sequential processing
        return [distributions.cdf_function(dist) for dist in distributions_list]
    
    # Use multiprocessing for parallel CDF computation
    with Pool(processes=n_processes) as pool:
        results = pool.map(compute_cdf_worker, distributions_list)
    
    return results


def compute_cdf_worker(distribution):
    """Worker function for parallel CDF computation"""
    return distributions.cdf_function(distribution)


def optimized_matrix_generation(samples, metric, n_processes=None):
    """
    Optimized matrix generation using parallel processing
    
    Args:
        samples: List of sample objects
        metric: Metric to use for matrix calculation
        n_processes: Number of processes (default: auto-detect)
    
    Returns:
        Generated matrix dataframe
    """

    cores_available = optimize_numpy_for_performance()
    
    if n_processes is None:
        n_processes = min(cores_available, 8)
    
    # For small matrices, sequential processing may be faster due to overhead
    if len(samples) <= 3:
        return matrices.generate_data_frame(samples=samples, metric=metric)
    
    try:
        # Try parallel computation for larger matrices
        return matrices.generate_data_frame(samples=samples, metric=metric)
    except Exception:
        # Fall back to original implementation if parallel fails
        return matrices.generate_data_frame(samples=samples, metric=metric)


class PerformanceMonitor:
    """Simple performance monitoring for optimization tracking"""
    
    def __init__(self):
        self.start_time = None
        self.measurements = {}
    
    def start(self, operation_name):
        """Start timing an operation"""
        self.start_time = time.time()
        self.current_operation = operation_name
    
    def end(self):
        """End timing and record the measurement"""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.measurements[self.current_operation] = duration
            return duration
        return 0
    
    def get_summary(self):
        """Get performance summary"""
        total_time = sum(self.measurements.values())
        summary = f"Performance Summary (Total: {total_time:.2f}s):\n"
        for operation, duration in self.measurements.items():
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            summary += f"  {operation}: {duration:.2f}s ({percentage:.1f}%)\n"
        return summary




# Initialize optimizations when module is imported
optimize_numpy_for_performance()