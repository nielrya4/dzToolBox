"""
Optimized Monte Carlo implementation for multi-core processing
Designed for HP Z800 with 12 cores
"""
import numpy as np
import random
from multiprocessing import Pool, cpu_count
from functools import partial
import os
from dz_lib.univariate import metrics
from dz_lib.univariate.distributions import Distribution


def optimize_numpy_threads():
    """Configure numpy to use multiple threads for mathematical operations"""
    from .math_optimizations import optimize_numpy_for_performance
    return optimize_numpy_for_performance()


class OptimizedUnmixingTrial:
    """Optimized version of UnmixingTrial with vectorized operations"""
    
    def __init__(self, sink_line: np.ndarray, source_lines: list, metric: str = "cross_correlation"):
        self.sink_line = np.asarray(sink_line, dtype=np.float32)
        self.source_lines = [np.asarray(line, dtype=np.float32) for line in source_lines]
        self.metric = metric
        self.n_sources = len(source_lines)
        
    def run_trial(self):
        """Execute a single trial with optimized numpy operations"""
        # Generate normalized random weights
        rands = np.random.random(self.n_sources).astype(np.float32)
        rands = rands / np.sum(rands)
        
        # Vectorized model calculation
        model_line = np.zeros_like(self.sink_line, dtype=np.float32)
        for j, source_line in enumerate(self.source_lines):
            model_line += source_line * rands[j]
        
        # Calculate metric
        if self.metric == "cross_correlation":
            val = metrics.r2(self.sink_line, model_line)
        elif self.metric == "ks":
            val = metrics.ks(self.sink_line, model_line)
        elif self.metric == "kuiper":
            val = metrics.kuiper(self.sink_line, model_line)
        else:
            raise ValueError(f"Unknown metric '{self.metric}'")
        
        return rands, model_line, val


def run_batch_trials(args):
    """Run a batch of trials in parallel worker process"""
    sink_line, source_lines, metric, batch_size, seed_offset = args
    
    # Set unique random seed for this worker
    np.random.seed(seed_offset)
    random.seed(seed_offset)
    
    # Convert to numpy arrays in worker process
    sink_array = np.asarray(sink_line, dtype=np.float32)
    source_arrays = [np.asarray(line, dtype=np.float32) for line in source_lines]
    
    trial_runner = OptimizedUnmixingTrial(sink_array, source_arrays, metric)
    
    results = []
    for _ in range(batch_size):
        rands, model_line, val = trial_runner.run_trial()
        results.append((rands.copy(), model_line.copy(), val))
    
    return results


def monte_carlo_model_optimized(
    sink_distribution: Distribution, 
    source_distributions: list[Distribution], 
    n_trials: int = 10000, 
    metric: str = "cross_correlation",
    n_processes: int = None
):
    """
    Optimized Monte Carlo model using multiprocessing when beneficial
    
    Args:
        sink_distribution: Target Distribution object
        source_distributions: List of source Distribution objects
        n_trials: Number of Monte Carlo trials
        metric: Metric to optimize ("cross_correlation", "ks", "kuiper")
        n_processes: Number of processes (default: auto-decide based on workload)
    
    Returns:
        source_contributions, source_std, top_distributions
    """
    # Configure numpy for optimal performance
    cores_available = optimize_numpy_threads()
    
    # Extract y_values from Distribution objects and convert to numpy arrays
    sink_line = np.asarray(sink_distribution.y_values, dtype=np.float32)
    source_lines = [np.asarray(dist.y_values, dtype=np.float32) for dist in source_distributions]
    
    # Smart decision: only use multiprocessing for large workloads
    # Multiprocessing overhead isn't worth it for small problems
    use_multiprocessing = n_trials >= 2000 and len(source_distributions) >= 2
    
    if not use_multiprocessing or n_processes == 1:
        # Use optimized sequential processing for small workloads
        return monte_carlo_sequential_optimized(sink_line, source_lines, n_trials, metric, sink_distribution.x_values)
    
    if n_processes is None:
        n_processes = min(cores_available, 8)  # Conservative for better efficiency
    
    # Calculate batch size per process
    batch_size = max(100, n_trials // n_processes)  # Minimum batch size to reduce overhead
    total_batches = min(n_processes, n_trials // batch_size)
    
    # Prepare arguments for parallel processing
    args_list = []
    trials_assigned = 0
    for i in range(total_batches):
        current_batch_size = batch_size
        if i == total_batches - 1:
            current_batch_size = n_trials - trials_assigned
        
        if current_batch_size > 0:
            seed_offset = i * 1000
            args_list.append((sink_line, source_lines, metric, current_batch_size, seed_offset))
            trials_assigned += current_batch_size
    
    # Run parallel processing
    with Pool(processes=len(args_list)) as pool:
        batch_results = pool.map(run_batch_trials, args_list)
    
    # Collect all results
    all_trials = []
    for batch_result in batch_results:
        all_trials.extend(batch_result)
    
    # Sort trials by metric value
    if metric == "cross_correlation":
        sorted_trials = sorted(all_trials, key=lambda x: x[2], reverse=True)
    elif metric in ["ks", "kuiper"]:
        sorted_trials = sorted(all_trials, key=lambda x: x[2], reverse=False)
    else:
        raise ValueError(f"Unknown metric '{metric}'")
    
    # Get top 10 trials
    top_trials = sorted_trials[:10]
    top_lines = [trial[1] for trial in top_trials]
    random_configurations = [trial[0] for trial in top_trials]
    
    # Convert top lines back to Distribution objects
    x_values = sink_distribution.x_values
    top_distributions = [Distribution(f"Top_Trial_{i+1}", x_values, y_values.tolist()) 
                        for i, y_values in enumerate(top_lines)]
    
    # Calculate statistics using vectorized operations
    random_configs_array = np.array(random_configurations)
    source_contributions = np.mean(random_configs_array, axis=0) * 100
    source_std = np.std(random_configs_array, axis=0) * 100
    
    return source_contributions, source_std, top_distributions


def monte_carlo_sequential_optimized(sink_line, source_lines, n_trials, metric, x_values):
    """Optimized sequential Monte Carlo for small workloads"""
    trial_runner = OptimizedUnmixingTrial(sink_line, source_lines, metric)
    
    results = []
    for _ in range(n_trials):
        rands, model_line, val = trial_runner.run_trial()
        results.append((rands, model_line, val))
    
    # Sort trials by metric value  
    if metric == "cross_correlation":
        sorted_trials = sorted(results, key=lambda x: x[2], reverse=True)
    elif metric in ["ks", "kuiper"]:
        sorted_trials = sorted(results, key=lambda x: x[2], reverse=False)
    else:
        raise ValueError(f"Unknown metric '{metric}'")
    
    # Get top 10 trials
    top_trials = sorted_trials[:10]
    top_lines = [trial[1] for trial in top_trials]
    random_configurations = [trial[0] for trial in top_trials]
    
    # Convert top lines back to Distribution objects
    top_distributions = [Distribution(f"Top_Trial_{i+1}", x_values, y_values.tolist()) 
                        for i, y_values in enumerate(top_lines)]
    
    # Calculate statistics
    random_configs_array = np.array(random_configurations)
    source_contributions = np.mean(random_configs_array, axis=0) * 100
    source_std = np.std(random_configs_array, axis=0) * 100
    
    return source_contributions, source_std, top_distributions


def benchmark_monte_carlo(sink_y_values, sources_y_values, n_trials=1000):
    """
    Benchmark function to compare single-core vs multi-core performance
    """
    import time
    from dz_lib.univariate.unmix import monte_carlo_model as original_monte_carlo
    
    print(f"Benchmarking Monte Carlo with {n_trials} trials...")
    
    # Benchmark original single-core version
    start_time = time.time()
    orig_contrib, orig_std, orig_lines = original_monte_carlo(
        sink_y_values, sources_y_values, n_trials, "cross_correlation"
    )
    single_core_time = time.time() - start_time
    
    # Benchmark optimized multi-core version
    start_time = time.time()
    opt_contrib, opt_std, opt_lines = monte_carlo_model_optimized(
        sink_y_values, sources_y_values, n_trials, "cross_correlation"
    )
    multi_core_time = time.time() - start_time
    
    speedup = single_core_time / multi_core_time
    
    print(f"Single-core time: {single_core_time:.2f}s")
    print(f"Multi-core time: {multi_core_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    return speedup


# Monkey patch to replace original function
def patch_monte_carlo_function():
    """Replace the original monte_carlo_model with optimized version"""
    import dz_lib.univariate.unmix as unmix_module
    
    # Store original function for benchmarking
    unmix_module.monte_carlo_model_original = unmix_module.monte_carlo_model
    
    # Replace with optimized version
    unmix_module.monte_carlo_model = monte_carlo_model_optimized
    
    print(f"Monte Carlo optimized for {cpu_count()} cores")


if __name__ == "__main__":
    # Test the optimization
    patch_monte_carlo_function()
    
    # Generate test data
    np.random.seed(42)
    sink_data = np.random.random(1000)
    source_data = [np.random.random(1000) for _ in range(3)]
    
    # Run benchmark
    speedup = benchmark_monte_carlo(sink_data, source_data, 10000)
    print(f"Achieved {speedup:.1f}x speedup on {cpu_count()} cores")