"""
Maximum Depositional Age (MDA) Module

This module provides convention-explicit implementations of MDA metrics for
detrital geochronology. Every published MDA metric is a family of algorithms,
not one algorithm - this module makes every choice explicit and inspectable.

Based on mda_harmonized by [author], which was written from the published
method descriptions with no code copied from existing MDA packages.

Methods included:
- YSG: Youngest Single Grain
- YC1s/YC2s: Youngest Grain Cluster at 1σ/2σ
- Y3Za: Youngest 3 Zircons (Average, no overlap required)
- Y3Zo: Youngest 3 Zircons (Overlap required)
- YPP: Youngest Graphical Peak
- YGF: Youngest Gaussian Fit
- YSP: Youngest Statistical Population
- Tau: Tau Method
- YDZ: Youngest Detrital Zircon (Monte Carlo)
- MLA: Maximum Likelihood Age

Presets:
- python_toolset: Reproduces detritalPy conventions
- matlab_toolset: Reproduces DZmda conventions
- harmonized: Recommended defaults with explicit reasoning

Usage:
    from dz_lib.univariate.mda import Sample, all_metrics, to_table

    # Create sample with explicit sigma level
    smp = Sample(ages, errs, sigma_in=1, name="MySample")

    # Run all metrics with a preset
    results = all_metrics(smp, preset="harmonized")
    print(to_table(results))

    # Individual metrics with custom conventions
    from dz_lib.univariate.mda import ysg, yc, ysp, mla
    print(yc(smp, k_sigma=2.0, min_n=3, anchor="youngest", mutual=True))
"""

# Import the new harmonized implementation
from .core import (
    # Containers
    Sample,
    MDAResult,

    # Core statistics
    weighted_mean,
    pdp,
    kde,

    # Metrics
    ysg,
    yc,
    y3za,
    y3zo,
    ypp,
    ygf,
    ysp,
    tau,
    ydz,
    mla,

    # Batch processing
    all_metrics,
    PRESETS,
    to_table,
)

# Import legacy visualization functions that aren't in core
from .mla import radial_plot

# Legacy adapter imports for backwards compatibility
from .adapters import (
    # Legacy function names
    youngest_single_grain,
    youngest_cluster_1s,
    youngest_cluster_2s,
    youngest_3_zircons,
    youngest_3_zircons_overlap,
    youngest_graphical_peak,
    youngest_statistical_population,
    tau_method,
    youngest_gaussian_fit,
    maximum_likelihood_age,

    # Legacy utilities
    get_weighted_mean,
    get_youngest_cluster,
    count_bins_around_peak,

    # Legacy visualization
    ranked_ages_plot,
    comparison_graph,
    comparison_table,
)

__all__ = [
    # New API (from core)
    'Sample',
    'MDAResult',
    'weighted_mean',
    'pdp',
    'kde',
    'ysg',
    'yc',
    'y3za',
    'y3zo',
    'ypp',
    'ygf',
    'ysp',
    'tau',
    'ydz',
    'mla',
    'all_metrics',
    'PRESETS',
    'to_table',

    # Visualization
    'radial_plot',

    # Legacy API (backwards compatibility)
    'youngest_single_grain',
    'youngest_cluster_1s',
    'youngest_cluster_2s',
    'youngest_3_zircons',
    'youngest_3_zircons_overlap',
    'youngest_graphical_peak',
    'youngest_statistical_population',
    'tau_method',
    'youngest_gaussian_fit',
    'maximum_likelihood_age',
    'get_weighted_mean',
    'get_youngest_cluster',
    'count_bins_around_peak',
    'ranked_ages_plot',
    'comparison_graph',
    'comparison_table',
]
