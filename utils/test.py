# measures.py by Kurt Sundell. Interpreted by Ryan Nielsen to work directly with detrital zircon samples.
import numpy as np


# KS Test (Massey, 1951) is the max absolute difference btw 2 CDF curves
def ks(y1_values, y2_values):
    data1, data2 = np.ma.asarray(y1_values), np.ma.asarray(y2_values)
    n1, n2 = (data1.count(), data2.count())
    mix = np.ma.concatenate((data1.compressed(), data2.compressed()))
    mix_sort = mix.argsort(kind='mergesort')
    csum = np.where(mix_sort < n1, 1. / n1, -1. / n2).cumsum()
    ks_test_d = max(np.abs(csum))
    return ks_test_d


# Kuiper test (Kuiper, 1960) is the sum of the max difference of CDF1 - CDF2 and CDF2 - CDF1
def kuiper(y1_values, y2_values):
    data1, data2 = np.ma.asarray(y1_values), np.ma.asarray(y2_values)
    n1, n2 = data1.count(), data2.count()
    mix = np.ma.concatenate((data1.compressed(), data2.compressed()))
    mix_sort = mix.argsort(kind='mergesort')
    csum = np.where(mix_sort < n1, 1. / n1, -1. / n2).cumsum()
    kuiper_test_v = max(csum) + max(csum * -1)
    return kuiper_test_v


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
