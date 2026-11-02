"""
Unmixing entry point used by the app.

The model itself lives in dz_lib.univariate.unmix.monte_carlo_model; it scores all trials
with vectorised numpy, which is faster than the previous per-trial multiprocessing pool.
"""
from dz_lib.univariate.distributions import Distribution
from dz_lib.univariate.unmix import monte_carlo_model


def optimize_numpy_threads():
    """Configure numpy to use multiple threads for mathematical operations"""
    from .math_optimizations import optimize_numpy_for_performance
    return optimize_numpy_for_performance()


def monte_carlo_model_optimized(
    sink_distribution: Distribution,
    source_distributions: list[Distribution],
    n_trials: int = 10000,
    metric: str = "cross_correlation",
    n_processes: int = None,
    refine: bool = True,
    sampler: str = "dirichlet"
):
    """
    Monte Carlo unmixing, optionally refined by constrained optimisation.

    Args:
        sink_distribution: Target Distribution object
        source_distributions: List of source Distribution objects
        n_trials: Number of Monte Carlo trials
        metric: Metric to optimize ("cross_correlation", "ks", "kuiper")
        n_processes: Unused; kept for backward compatibility
        refine: Refine the best trials by constrained optimisation (see monte_carlo_model)
        sampler: How trial proportions are drawn: "dirichlet" (uniform) or "dzmix"

    Returns:
        source_contributions, source_std, top_distributions
    """
    optimize_numpy_threads()
    return monte_carlo_model(sink_distribution, source_distributions, n_trials=n_trials, metric=metric,
                             refine=refine, sampler=sampler)
