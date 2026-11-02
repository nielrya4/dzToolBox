# measures.py by Kurt Sundell, interpreted by Ryan Nielsen to work directly with detrital zircon samples.
import numpy as np

# KS Test (Massey, 1951) is the max absolute difference btw 2 CDF curves
def ks(y1_values, y2_values):
    d_val = max(abs(y1_values - y2_values))
    return d_val

# Kuiper test (Kuiper, 1960) is the sum of the max difference of CDF1 - CDF2 and CDF2 - CDF1
def kuiper(y1_values, y2_values):
    v_val = max(y1_values - y2_values) + max(y2_values - y1_values)
    return v_val

# Similarity (Gehrels, 2000) is the sum of the geometric mean of each point along x for two PDPs or KDEs
def similarity(y1_values, y2_values):
    similarity = np.sum(np.sqrt(y1_values * y2_values))
    return similarity

# Likeness (Satkoski et al., 2013) is the complement to Mismatch (Amidon et al., 2005) and is the sum of the
# absolute difference divided by 2 for every pair of points along x for two PDPs or KDEs
def likeness(y1_values, y2_values):
    likeness = 1 - np.sum(abs(y1_values - y2_values)) / 2
    return likeness

# Cross-correlation is the coefficient of determination (R squared),
# the simple linear regression between two PDPs or KDEs
def r2(y1_values, y2_values):
    correlation_matrix = np.corrcoef(y1_values, y2_values)
    correlation_xy = correlation_matrix[0, 1]
    cross_correlation = correlation_xy ** 2
    return cross_correlation

# The KS and Kuiper tests are defined on the empirical CDFs of the ages themselves (as in
# DZstats, DZmds, DZmix and detritalPy), not on the CDF of a smoothed PDP or KDE.
def ecdf(ages, x_values):
    """Empirical CDF of `ages` evaluated at `x_values` (right-continuous)."""
    ages = np.sort(np.asarray(ages, dtype=float))
    return np.searchsorted(ages, np.asarray(x_values, dtype=float), side="right") / ages.size


def _ecdf_differences(ages1, ages2):
    pooled = np.concatenate([np.asarray(ages1, dtype=float), np.asarray(ages2, dtype=float)])
    return ecdf(ages1, pooled) - ecdf(ages2, pooled)


def ks_ages(ages1, ages2):
    """Two-sample KS D from the raw ages (matches MATLAB kstest2 and scipy ks_2samp)."""
    return float(np.max(np.abs(_ecdf_differences(ages1, ages2))))


def kuiper_ages(ages1, ages2):
    """Two-sample Kuiper V from the raw ages (matches kuipertest2c in DZstats)."""
    diff = _ecdf_differences(ages1, ages2)
    return float(max(np.max(diff), 0.0) + max(np.max(-diff), 0.0))

def dis_similarity(y1_values, y2_values):
    return float(1 - similarity(y1_values, y2_values))

def dis_ks(y1_values, y2_values):
    return float(1-ks(y1_values, y2_values))

def dis_kuiper(y1_values, y2_values):
    return float(1 - kuiper(y1_values, y2_values))

def dis_likeness(y1_values, y2_values):
    return float(1 - likeness(y1_values, y2_values))

def dis_r2(y1_values, y2_values):
    return float(1 - r2(y1_values, y2_values))
