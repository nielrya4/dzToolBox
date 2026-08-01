"""
Legacy API Adapters for MDA Module

This module provides backwards-compatible wrappers around the new harmonized
MDA implementation. These functions accept the legacy `Grain` objects and
return results in the legacy format.

For new code, prefer using the new API directly:
    from dz_lib.univariate.mda import Sample, ysg, yc, all_metrics

Legacy usage (still supported):
    from dz_lib.univariate.mda import youngest_single_grain, youngest_cluster_1s
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

from dz_lib.univariate.data import Grain, Sample as LegacySample
from dz_lib.univariate import distributions
from dz_lib import config

from .core import (
    Sample as MdaSample,
    ysg as _ysg,
    yc as _yc,
    y3za as _y3za,
    y3zo as _y3zo,
    ypp as _ypp,
    ygf as _ygf,
    ysp as _ysp,
    tau as _tau,
    ydz as _ydz,
    mla as _mla,
    weighted_mean as _weighted_mean,
)


def _grains_to_sample(grains: List[Grain], sigma_in: int = None) -> MdaSample:
    """Convert legacy Grain list to new Sample object."""
    if sigma_in is None:
        sigma_in = config.get_sigma_in()

    ages = np.array([g.age for g in grains])
    errs = np.abs(np.array([g.uncertainty for g in grains]))  # Use absolute value

    # Filter out grains with non-finite values
    valid_mask = np.isfinite(ages) & np.isfinite(errs)
    ages = ages[valid_mask]
    errs = errs[valid_mask]

    if len(ages) == 0:
        raise ValueError("No valid grains found")

    # Replace zero uncertainties with a small default (1% of age or 1 Ma, whichever is larger)
    zero_mask = errs <= 0
    if np.any(zero_mask):
        default_errs = np.maximum(ages[zero_mask] * 0.01, 1.0)
        errs[zero_mask] = default_errs

    return MdaSample(ages, errs, sigma_in=sigma_in)


def _result_to_grain(result, sigma_out: int = None) -> Grain:
    """Convert MDAResult to legacy Grain object."""
    if sigma_out is None:
        sigma_out = config.get_sigma_out()

    # Convert 1-sigma uncertainty to requested sigma level
    unc = result.unc_1s * sigma_out if np.isfinite(result.unc_1s) else float('nan')
    return Grain(age=result.mda, uncertainty=unc)


# =============================================================================
# Legacy MDA Functions
# =============================================================================

def youngest_single_grain(
    grains: List[Grain],
    sigma_in: int = None
) -> Tuple[Grain, float]:
    """
    Youngest Single Grain (YSG).

    Legacy wrapper - for new code use: ysg(sample)
    """
    sample = _grains_to_sample(grains, sigma_in)
    result = _ysg(sample, rank_by="age+1s")
    grain = _result_to_grain(result)
    return grain, 1.0


def youngest_cluster_1s(
    grains: List[Grain],
    min_cluster_size: int = 2,
    include_yg: bool = True,
    sigma_in: int = None
) -> Tuple[Grain, int, float]:
    """
    Youngest Grain Cluster at 1σ (YC1s).

    Legacy wrapper - for new code use: yc(sample, k_sigma=1.0, min_n=2)
    """
    sample = _grains_to_sample(grains, sigma_in)
    anchor = "youngest" if include_yg else "scan"
    result = _yc(sample, k_sigma=1.0, min_n=min_cluster_size,
                 rank_by="age+1s", anchor=anchor)
    grain = _result_to_grain(result)
    return grain, result.n_used, result.mswd


def youngest_cluster_2s(
    grains: List[Grain],
    min_cluster_size: int = 3,
    include_yg: bool = True,
    sigma_in: int = None
) -> Tuple[Grain, int, float]:
    """
    Youngest Grain Cluster at 2σ (YC2s).

    Legacy wrapper - for new code use: yc(sample, k_sigma=2.0, min_n=3)
    """
    sample = _grains_to_sample(grains, sigma_in)
    anchor = "youngest" if include_yg else "scan"
    result = _yc(sample, k_sigma=2.0, min_n=min_cluster_size,
                 rank_by="age+2s", anchor=anchor)
    grain = _result_to_grain(result)
    return grain, result.n_used, result.mswd


def youngest_3_zircons(
    grains: List[Grain],
    sigma_in: int = None
) -> Tuple[Grain, int, float]:
    """
    Youngest 3 Zircons Average (Y3Za).

    Legacy wrapper - for new code use: y3za(sample, n_grains=3)
    """
    sample = _grains_to_sample(grains, sigma_in)
    result = _y3za(sample, n_grains=3, rank_by="age")
    grain = _result_to_grain(result)
    return grain, result.n_used, result.mswd


def youngest_3_zircons_overlap(
    grains: List[Grain],
    sigma: int = 2,
    sigma_in: int = None
) -> Tuple[Grain, int, float]:
    """
    Youngest 3 Zircons with Overlap (Y3Zo).

    Legacy wrapper - for new code use: y3zo(sample, n_grains=3, k_sigma=2.0)
    """
    sample = _grains_to_sample(grains, sigma_in)
    result = _y3zo(sample, n_grains=3, k_sigma=float(sigma))
    grain = _result_to_grain(result)
    return grain, result.n_used, result.mswd


def youngest_graphical_peak(
    grains: List[Grain],
    min_cluster_size: int = 2,
    threshold: float = 0.01,
    min_dist: float = 1.0,
    x_min: float = 0,
    x_max: float = 4500,
    sigma_in: int = None
) -> float:
    """
    Youngest Graphical Peak (YPP).

    Legacy wrapper - for new code use: ypp(sample)
    """
    sample = _grains_to_sample(grains, sigma_in)
    result = _ypp(sample, min_n=min_cluster_size, prominence_frac=threshold,
                  min_sep_myr=min_dist)
    return result.mda if np.isfinite(result.mda) else float('nan')


def youngest_statistical_population(
    grains: List[Grain],
    min_cluster_size: int = 2,
    mswd_threshold: float = 1.0,
    sigma: float = 1.0,
    add_uncertainty: bool = False,
    sigma_in: int = None
) -> Tuple[Grain, int, float]:
    """
    Youngest Statistical Population (YSP).

    Legacy wrapper - for new code use: ysp(sample)
    """
    sample = _grains_to_sample(grains, sigma_in)
    rank_by = "age+1s" if add_uncertainty else "age"
    result = _ysp(sample, min_n=min_cluster_size, rank_by=rank_by,
                  target_mswd=mswd_threshold, entry_rule="global")
    grain = _result_to_grain(result)
    return grain, result.n_used, result.mswd


def tau_method(
    grains: List[Grain],
    mode_req: int = 3,
    thres: float = 0.01,
    min_dist: int = 1,
    x1: float = 0,
    x2: float = 4500,
    sigma_in: int = None
) -> Tuple[Grain, int, float]:
    """
    Tau Method.

    Legacy wrapper - for new code use: tau(sample)
    """
    sample = _grains_to_sample(grains, sigma_in)
    result = _tau(sample, min_n=mode_req, prominence_frac=thres, bounds="troughs")
    grain = _result_to_grain(result)
    return grain, result.n_used, result.mswd


def youngest_gaussian_fit(
    grains: List[Grain],
    x_min: float = 0,
    x_max: float = 4500,
    sigma_in: int = None
) -> Tuple[Grain, distributions.Distribution]:
    """
    Youngest Gaussian Fit (YGF).

    Legacy wrapper - for new code use: ygf(sample)
    """
    sample = _grains_to_sample(grains, sigma_in)
    result = _ygf(sample, min_n=3)

    grain = _result_to_grain(result)

    # Generate fitted distribution for visualization
    # This is approximate - the new implementation doesn't return the full curve
    if np.isfinite(result.mda) and np.isfinite(result.unc_1s):
        mu = result.mda
        sigma = result.unc_1s
        x = np.linspace(x_min, x_max, int((x_max - x_min) * 10))
        y = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        y = y / (sigma * np.sqrt(2 * np.pi))  # Normalize
        fitted_distro = distributions.Distribution(
            f"Youngest Gaussian Fit\nMean: {mu:.2f} Ma\n1σ: {sigma:.2f}",
            x, y
        )
    else:
        fitted_distro = distributions.Distribution("No fit", np.array([]), np.array([]))

    return grain, fitted_distro


def maximum_likelihood_age(
    grains: List[Grain],
    n_params: int = 4,
    verbose: bool = False,
    sigma_in: int = None
) -> Tuple[Grain, int, float]:
    """
    Maximum Likelihood Age (MLA).

    Legacy wrapper - for new code use: mla(sample)
    """
    sample = _grains_to_sample(grains, sigma_in)
    result = _mla(sample, log=True, n_starts=12, seed=0)
    grain = _result_to_grain(result)
    return grain, result.n_used, float('nan')


# =============================================================================
# Legacy Utility Functions
# =============================================================================

def get_weighted_mean(
    grains: List[Grain],
    confidence_level: float = 0.95,
    sigma_in: int = None
) -> Tuple[float, float, float]:
    """
    Calculate weighted mean of grain ages.

    Legacy wrapper - for new code use: weighted_mean(ages, s1)
    """
    if sigma_in is None:
        sigma_in = config.get_sigma_in()

    ages = np.array([g.age for g in grains])
    errs = np.array([g.uncertainty for g in grains])
    s1 = errs / sigma_in  # Normalize to 1-sigma

    result = _weighted_mean(ages, s1)

    # Convert to requested confidence level
    # Legacy used scipy.stats.norm.ppf which gives 1.96 for 95%
    import scipy.stats as stats
    z = stats.norm.ppf(confidence_level + (1 - confidence_level) / 2.)
    uncertainty = z * result['se1s']

    return result['wm'], uncertainty, result['mswd']


def get_youngest_cluster(
    grains: List[Grain],
    min_cluster_size: int,
    add_uncertainty: bool = False,
    contiguous: bool = True,
    sigma_in: int = None
) -> List[Grain]:
    """
    Find the youngest cluster of overlapping grains.

    Legacy function - consider using yc() for more control.
    """
    if sigma_in is None:
        sigma_in = config.get_sigma_in()

    # Sort grains
    if add_uncertainty:
        sorted_grains = sorted(grains, key=lambda g: g.age + g.uncertainty)
    else:
        sorted_grains = sorted(grains, key=lambda g: g.age)

    # Convert to 1-sigma for overlap testing
    ages = np.array([g.age for g in sorted_grains])
    s1 = np.array([g.uncertainty / sigma_in for g in sorted_grains])

    hi = ages + s1
    lo = ages - s1

    for i in range(len(sorted_grains)):
        overlaps = [lo[j] < hi[i] for j in range(i, len(sorted_grains))]

        if not contiguous:
            if sum(overlaps) >= min_cluster_size:
                return [sorted_grains[j] for j in range(i, len(sorted_grains)) if overlaps[j-i]]
        else:
            # Find first non-overlap
            try:
                first_false = overlaps.index(False)
            except ValueError:
                first_false = len(overlaps)

            if first_false >= min_cluster_size:
                return sorted_grains[i:i + first_false]

    return []


def count_bins_around_peak(
    peak_age: float,
    distribution: distributions.Distribution,
    window: float = 1.0
) -> int:
    """Count bins within a window around a peak."""
    return sum(1 for x in distribution.x_values if abs(x - peak_age) <= window / 2)


# =============================================================================
# Legacy Visualization Functions
# =============================================================================

def ranked_ages_plot(
    grains: List[Grain],
    x_min: float = 0,
    x_max: float = 4500,
    sort_with_uncertainty: bool = True,
    legend: bool = True,
    title: str = None,
    font_path: str = None,
    font_size: float = 12,
    fig_width: float = 9,
    fig_height: float = 7,
    color_1s: str = "black",
    color_2s: str = "cornflowerblue",
):
    """Create a ranked ages plot."""
    if sort_with_uncertainty:
        sorted_grains = sorted(grains, key=lambda g: g.age + abs(g.uncertainty) * 2)
    else:
        sorted_grains = sorted(grains, key=lambda g: g.age)

    ages = np.array([g.age for g in sorted_grains])
    uncertainties = np.array([abs(g.uncertainty) for g in sorted_grains])
    ranks = np.arange(len(sorted_grains))

    # Filter to range
    mask = (ages > x_min) & (ages < x_max)
    ages = ages[mask]
    uncertainties = uncertainties[mask]
    ranks = ranks[mask]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    ax.scatter(ages, ranks, facecolors='white', edgecolors="k", marker='d', s=100, zorder=10)
    ax.hlines(ranks, ages - 2 * uncertainties, ages + 2 * uncertainties,
              color=color_2s, linewidth=4, label='2σ')
    ax.hlines(ranks, ages - uncertainties, ages + uncertainties,
              color=color_1s, linewidth=4, label='1σ')

    if font_path:
        font = fm.FontProperties(fname=font_path)
    else:
        font = None

    ax.set_xlabel("Age (Ma)", fontsize=font_size, fontproperties=font)
    ax.set_ylabel("Ranked Grains", fontsize=font_size, fontproperties=font)

    if title:
        ax.set_title(title, fontsize=font_size * 1.5, fontproperties=font)
    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)

    ax.invert_yaxis()
    ax.set_xlim(x_min, x_max)
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 1])
    plt.close()

    return fig


def comparison_graph(
    grains: List[Grain],
    title: str = None,
    legend: bool = True,
    font_path: str = None,
    font_size: float = 12,
    fig_width: float = 9,
    fig_height: float = 7,
    color_1s: str = "black",
    color_2s: str = "cornflowerblue",
):
    """Create a comparison graph of all MDA methods."""
    # Calculate all MDAs
    ysg_grain, _ = youngest_single_grain(grains)
    ypp_age = youngest_graphical_peak(grains)
    ypp_grain = Grain(ypp_age, float('nan'))
    ygf_grain, _ = youngest_gaussian_fit(grains)
    ygc1s_grain, _, _ = youngest_cluster_1s(grains)
    ygc2s_grain, _, _ = youngest_cluster_2s(grains)
    y3zo_grain, _, _ = youngest_3_zircons_overlap(grains)
    y3za_grain, _, _ = youngest_3_zircons(grains)
    tau_grain, _, _ = tau_method(grains)
    ysp_grain, _, _ = youngest_statistical_population(grains)
    mla_grain, _, _ = maximum_likelihood_age(grains)

    methods = ['YSG', 'YPP', 'YGF', 'YGC1s', 'YGC2s', 'Y3ZO', 'Y3Za', 'TAU', 'YSP', 'MLA']
    result_grains = [ysg_grain, ypp_grain, ygf_grain, ygc1s_grain, ygc2s_grain,
                     y3zo_grain, y3za_grain, tau_grain, ysp_grain, mla_grain]
    ages = [g.age for g in result_grains]
    uncertainties = [g.uncertainty for g in result_grains]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    x = np.arange(len(methods))

    for i in range(len(methods)):
        if np.isfinite(uncertainties[i]):
            ax.vlines(x[i], ages[i] - uncertainties[i] * 2, ages[i] + uncertainties[i] * 2,
                      color=color_2s, linewidth=5)
            ax.vlines(x[i], ages[i] - uncertainties[i], ages[i] + uncertainties[i],
                      color=color_1s, linewidth=5)

    ax.scatter(x, ages, color='white', edgecolor='black', s=100, zorder=3, marker='s')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_xlabel('Method', fontsize=font_size)
    ax.set_ylabel('Age (Ma)', fontsize=font_size)

    if title:
        if font_path:
            font_prop = fm.FontProperties(fname=font_path)
        else:
            font_prop = None
        ax.set_title(title, fontsize=font_size * 1.5, fontproperties=font_prop)

    if legend:
        legend_elements = [
            Line2D([0], [0], color=color_2s, lw=5, label='2s'),
            Line2D([0], [0], color=color_1s, lw=5, label='1s')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)

    fig.tight_layout()
    plt.close()

    return fig


def comparison_table(grains: List[Grain]) -> pd.DataFrame:
    """Create a comparison table of all MDA methods."""
    # Calculate all MDAs
    ysg_grain, ysg_n = youngest_single_grain(grains)
    ysg_mswd = float('nan')

    ypp_age = youngest_graphical_peak(grains)
    ypp_grain = Grain(ypp_age, float('nan'))
    ypp_n, ypp_mswd = float('nan'), float('nan')

    ygf_grain, _ = youngest_gaussian_fit(grains)
    ygf_n, ygf_mswd = float('nan'), float('nan')

    ygc1s_grain, ygc1s_n, ygc1s_mswd = youngest_cluster_1s(grains)
    ygc2s_grain, ygc2s_n, ygc2s_mswd = youngest_cluster_2s(grains)
    y3zo_grain, y3zo_n, y3zo_mswd = youngest_3_zircons_overlap(grains)
    y3za_grain, y3za_n, y3za_mswd = youngest_3_zircons(grains)
    tau_grain, tau_n, tau_mswd = tau_method(grains)
    ysp_grain, ysp_n, ysp_mswd = youngest_statistical_population(grains)
    mla_grain, mla_n, mla_mswd = maximum_likelihood_age(grains)

    methods = ['YSG', 'YPP', 'YGF', 'YGC1s', 'YGC2s', 'Y3ZO', 'Y3Za', 'TAU', 'YSP', 'MLA']
    result_grains = [ysg_grain, ypp_grain, ygf_grain, ygc1s_grain, ygc2s_grain,
                     y3zo_grain, y3za_grain, tau_grain, ysp_grain, mla_grain]
    ages = [g.age for g in result_grains]
    uncertainties = [g.uncertainty for g in result_grains]
    n_values = [ysg_n, ypp_n, ygf_n, ygc1s_n, ygc2s_n, y3zo_n, y3za_n, tau_n, ysp_n, mla_n]
    mswd_values = [ysg_mswd, ypp_mswd, ygf_mswd, ygc1s_mswd, ygc2s_mswd,
                   y3zo_mswd, y3za_mswd, tau_mswd, ysp_mswd, mla_mswd]

    data = {
        "MDA (Ma)": ages,
        "1s (Myr)": uncertainties,
        "2s (Myr)": [u * 2 if np.isfinite(u) else float('nan') for u in uncertainties],
        "n": n_values,
        "MSWD": mswd_values
    }

    df = pd.DataFrame(data, index=methods)
    df = df.rename_axis(columns="")

    return df
