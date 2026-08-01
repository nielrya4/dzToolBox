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
