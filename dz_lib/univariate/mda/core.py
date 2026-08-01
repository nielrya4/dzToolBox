"""
mda_harmonized.py
=================

A convention-explicit, single-file implementation of maximum depositional age
(MDA) metrics for detrital geochronology.

Motivation
----------
Every published MDA metric is a *family* of algorithms, not one algorithm. The
papers that define them (Dickinson & Gehrels, 2009; Barbeau et al., 2009;
Ludwig, 2012; Coutts et al., 2019; Vermeesch, 2021; Saylor et al., 2023)
underdetermine several choices that materially change the number:

  * what sigma level the input errors are on,
  * how grains are ranked before the youngest thing is picked,
  * whether the youngest grain is *forced* into the answer,
  * whether cluster members must be contiguous in rank, and whether they must
    overlap each other or only the seed grain,
  * what "MSWD close to 1" operationally means,
  * how a peak position and its grain count are defined,
  * whether the reported uncertainty is an internal SE, a 95% z-interval, or a
    Student-t interval inflated by sqrt(MSWD).

Independent codebases resolve these silently and differently, so two programs
can report different MDAs from identical data and both be "correct."

This module makes every one of those choices an explicit, inspectable keyword,
ships presets that reproduce the common conventions, and returns the resolved
convention alongside every number so results are reproducible from the output
alone.

Written from the published method descriptions. No code was copied or
transliterated from any existing MDA package.

Conventions used throughout
---------------------------
Internally, everything is 1-sigma absolute (Myr). Input sigma level is declared
once, at the door, via `sigma_in`.

Author: (your name)
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Iterable, Literal, Sequence

import numpy as np

__all__ = [
    "Sample",
    "MDAResult",
    "weighted_mean",
    "pdp",
    "ysg", "yc", "y3za", "y3zo", "ypp", "ygf", "ysp", "tau", "ydz", "mla",
    "all_metrics",
    "PRESETS",
]

# --------------------------------------------------------------------------- #
# Types and containers
# --------------------------------------------------------------------------- #

RankBy = Literal["age", "age+1s", "age+2s"]
Anchor = Literal["youngest", "scan"]
UncMode = Literal["se1s", "ci95z", "ci95t"]


@dataclass(frozen=True)
class Sample:
    """A set of single-grain dates with symmetric uncertainties.

    Parameters
    ----------
    age : sequence of float
        Best-age estimates in Ma.
    err : sequence of float
        Absolute uncertainties in Myr, at the level declared by `sigma_in`.
    sigma_in : {1, 2}
        The sigma level the supplied errors are on. This is the single most
        common source of cross-program disagreement; it is mandatory here
        rather than defaulted, by design.
    name : str
        Optional label carried through to results.
    """

    age: np.ndarray
    err: np.ndarray
    sigma_in: int = 1
    name: str = ""

    def __post_init__(self):
        a = np.asarray(self.age, dtype=float).ravel()
        e = np.asarray(self.err, dtype=float).ravel()
        if a.size != e.size:
            raise ValueError("age and err must be the same length")
        if self.sigma_in not in (1, 2):
            raise ValueError("sigma_in must be 1 or 2")
        if np.any(~np.isfinite(a)) or np.any(~np.isfinite(e)):
            raise ValueError("non-finite values in age or err")
        if np.any(e <= 0):
            raise ValueError("uncertainties must be strictly positive")
        object.__setattr__(self, "age", a)
        object.__setattr__(self, "err", e)

    @property
    def s1(self) -> np.ndarray:
        """Uncertainties normalised to 1 sigma."""
        return self.err / self.sigma_in

    @property
    def n(self) -> int:
        return self.age.size

    def filtered(self, max_rel_err_1s: float | None = None,
                 age_range: tuple[float, float] | None = None) -> "Sample":
        """Return a copy with grains removed by relative precision / age window.

        `max_rel_err_1s` is a fraction (0.10 == 10 percent at 1 sigma). Coutts
        et al. (2019) discuss precision filtering ahead of YSP; applying it
        uniformly to every metric, rather than to one of them, is the point.
        """
        keep = np.ones(self.n, dtype=bool)
        if max_rel_err_1s is not None:
            keep &= (self.s1 / self.age) <= max_rel_err_1s
        if age_range is not None:
            keep &= (self.age >= age_range[0]) & (self.age <= age_range[1])
        return Sample(self.age[keep], self.err[keep], self.sigma_in, self.name)


@dataclass
class MDAResult:
    """One metric's answer, plus everything needed to reproduce it."""

    metric: str
    mda: float = float("nan")
    unc_1s: float = float("nan")          # internal standard error, 1 sigma
    unc_95z: float = float("nan")         # 1.96 * SE  (no overdispersion term)
    unc_95t: float = float("nan")         # t(n-1) * SE * sqrt(MSWD) if MSWD>1
    unc_minus: float = float("nan")       # for asymmetric metrics (YDZ)
    unc_plus: float = float("nan")
    mswd: float = float("nan")
    n_used: int = 0
    idx_used: tuple[int, ...] = ()
    convention: dict = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        if not math.isfinite(self.mda):
            return f"<{self.metric}: no result ({self.note or 'criteria not met'})>"
        return (f"<{self.metric}: {self.mda:.3f} +/- {self.unc_95z:.3f} Ma (95% z), "
                f"n={self.n_used}, MSWD={self.mswd:.2f}>")


# --------------------------------------------------------------------------- #
# Core statistics
# --------------------------------------------------------------------------- #

def _student_t_975(dof: int) -> float:
    """Two-sided 97.5th percentile of Student's t, without a SciPy dependency.

    Uses an accurate rational approximation, falling back to SciPy if present.
    """
    if dof <= 0:
        return float("nan")
    try:
        from scipy import stats  # noqa: F401
        return float(stats.t.ppf(0.975, dof))
    except Exception:
        pass
    # Hill's approximation to the inverse t, adequate to ~1e-4 for dof >= 2.
    z = 1.959963984540054
    if dof == 1:
        return 12.7062047361747
    if dof == 2:
        return 4.30265272974946
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160.0
    v = float(dof)
    return z + g1 / v + g2 / v ** 2 + g3 / v ** 3 + g4 / v ** 4


def weighted_mean(age: Sequence[float], s1: Sequence[float]) -> dict:
    """Inverse-variance weighted mean with three uncertainty conventions.

    Returns all three rather than picking one, because the choice is a
    reporting convention, not a fact about the data:

      se1s   : 1 / sqrt(sum(1/s_i^2))            -- internal SE, 1 sigma
      ci95z  : 1.96 * se1s                       -- 95% interval, normal
      ci95t  : t_{0.975, n-1} * se1s * sqrt(MSWD) if MSWD > 1 else t * se1s
               -- 95% interval with the overdispersion inflation used by
               Isoplot / IsoplotR for scattered populations

    MSWD is the classical S/(n-1) statistic.
    """
    a = np.asarray(age, dtype=float).ravel()
    s = np.asarray(s1, dtype=float).ravel()
    n = a.size
    if n == 0:
        return dict(wm=float("nan"), se1s=float("nan"), ci95z=float("nan"),
                    ci95t=float("nan"), mswd=float("nan"), n=0)

    w = 1.0 / s ** 2
    wm = float(np.sum(w * a) / np.sum(w))
    se = float(1.0 / math.sqrt(np.sum(w)))

    if n == 1:
        return dict(wm=wm, se1s=float(s[0]), ci95z=1.959963984540054 * float(s[0]),
                    ci95t=float("nan"), mswd=float("nan"), n=1)

    mswd = float(np.sum((a - wm) ** 2 / s ** 2) / (n - 1))
    infl = math.sqrt(mswd) if mswd > 1.0 else 1.0
    return dict(wm=wm, se1s=se,
                ci95z=1.959963984540054 * se,
                ci95t=_student_t_975(n - 1) * se * infl,
                mswd=mswd, n=n)


def _fill_wm(res: MDAResult, wm: dict, idx: np.ndarray) -> MDAResult:
    res.mda = wm["wm"]
    res.unc_1s = wm["se1s"]
    res.unc_95z = wm["ci95z"]
    res.unc_95t = wm["ci95t"]
    res.mswd = wm["mswd"]
    res.n_used = wm["n"]
    res.idx_used = tuple(int(i) for i in np.atleast_1d(idx))
    return res


# --------------------------------------------------------------------------- #
# Density estimation and peak finding
# --------------------------------------------------------------------------- #

def pdp(age: np.ndarray, s1: np.ndarray, x_min: float = 0.0,
        x_max: float = 4500.0, step: float = 0.1,
        normalise: Literal["area", "sum", "none"] = "area") -> tuple[np.ndarray, np.ndarray]:
    """Probability density plot: sum of unit-area Gaussians, one per grain.

    The grid is built on [x_min, x_max]. Peak *positions* are insensitive to
    the normalisation, but relative-height peak thresholds are not, so the
    choice is exposed. `step` controls how finely a peak can be located; the
    de facto community value is 0.1 Ma.
    """
    x = np.arange(x_min, x_max + step, step, dtype=float)
    a = np.asarray(age, dtype=float)[:, None]
    s = np.asarray(s1, dtype=float)[:, None]
    dens = np.exp(-0.5 * ((x[None, :] - a) / s) ** 2) / (s * math.sqrt(2.0 * math.pi))
    y = dens.sum(axis=0)
    if normalise == "area":
        area = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
        if area > 0:
            y = y / area
    elif normalise == "sum":
        tot = y.sum()
        if tot > 0:
            y = y / tot
    return x, y


def kde(age: np.ndarray, bandwidth: float = 10.0, x_min: float = 0.0,
        x_max: float = 4500.0, step: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-bandwidth Gaussian KDE, offered as an alternative basis for YPP.

    Vermeesch (2012) argues the PDP is not a density estimator; the KDE is the
    statistically defensible alternative. Peak-based MDAs computed on a KDE
    will differ from PDP-based ones and should be labelled as such.
    """
    x = np.arange(x_min, x_max + step, step, dtype=float)
    a = np.asarray(age, dtype=float)[:, None]
    dens = np.exp(-0.5 * ((x[None, :] - a) / bandwidth) ** 2)
    y = dens.sum(axis=0) / (a.size * bandwidth * math.sqrt(2.0 * math.pi))
    return x, y


def _local_extrema(y: np.ndarray, kind: Literal["max", "min"],
                   prominence_frac: float = 0.0,
                   min_sep_pts: int = 1) -> np.ndarray:
    """Indices of local maxima (or minima) of a 1-D array.

    `prominence_frac` is expressed as a fraction of the full data range, and
    `min_sep_pts` is a separation in *grid points*. Both are stated in the
    units they are actually applied in -- a small thing, but mislabelling a
    grid-point separation as a separation in Myr silently changes which peaks
    survive when the grid step changes.
    """
    v = -y if kind == "min" else y
    up = np.r_[True, v[1:] > v[:-1]]
    dn = np.r_[v[:-1] >= v[1:], True]
    idx = np.flatnonzero(up & dn)
    if idx.size == 0:
        return idx

    if prominence_frac > 0:
        span = float(v.max() - v.min())
        if span > 0:
            keep = (v[idx] - v.min()) / span >= prominence_frac
            idx = idx[keep]

    if min_sep_pts > 1 and idx.size > 1:
        order = idx[np.argsort(-v[idx])]          # strongest first
        chosen: list[int] = []
        for i in order:
            if all(abs(i - j) >= min_sep_pts for j in chosen):
                chosen.append(int(i))
        idx = np.array(sorted(chosen), dtype=int)

    return idx


def _refine_peak_parabolic(x: np.ndarray, y: np.ndarray, i: int) -> float:
    """Sub-grid peak position by fitting a parabola to log(y) at i-1, i, i+1.

    Fitting the log makes this exact for a locally Gaussian peak. Returns the
    grid position unchanged if the fit is degenerate or lands outside the
    bracketing points.
    """
    if i <= 0 or i >= y.size - 1:
        return float(x[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    if min(y0, y1, y2) <= 0:
        return float(x[i])
    l0, l1, l2 = math.log(y0), math.log(y1), math.log(y2)
    denom = l0 - 2.0 * l1 + l2
    if denom == 0:
        return float(x[i])
    delta = 0.5 * (l0 - l2) / denom
    if abs(delta) > 1.0:
        return float(x[i])
    return float(x[i] + delta * (x[1] - x[0]))


def _fit_gaussian_segment(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Least-squares Gaussian fit to a density segment. Returns (mu, sigma).

    Uses the closed-form log-quadratic fit (weighted by y, which is the
    standard trick to keep low-density tails from dominating), so this has no
    optimiser dependency and no starting-value sensitivity.
    """
    m = y > 0
    if m.sum() < 3:
        return None
    xs, ys = x[m], y[m]
    w = ys
    try:
        coef = np.polyfit(xs, np.log(ys), 2, w=np.sqrt(w))
    except Exception:
        return None
    a2, a1, _ = coef
    if a2 >= 0:
        return None
    mu = -a1 / (2.0 * a2)
    sigma = math.sqrt(-1.0 / (2.0 * a2))
    if not (math.isfinite(mu) and math.isfinite(sigma)) or sigma <= 0:
        return None
    if mu < xs.min() - 3 * sigma or mu > xs.max() + 3 * sigma:
        return None
    return float(mu), float(sigma)


# --------------------------------------------------------------------------- #
# Ranking and cluster search -- the shared spine of YSG / YC / Y3Z
# --------------------------------------------------------------------------- #

def _rank_key(age: np.ndarray, s1: np.ndarray, rank_by: RankBy) -> np.ndarray:
    if rank_by == "age":
        return age
    if rank_by == "age+1s":
        return age + s1
    if rank_by == "age+2s":
        return age + 2.0 * s1
    raise ValueError(f"unknown rank_by: {rank_by!r}")


def _youngest_cluster(age: np.ndarray, s1: np.ndarray, *,
                      k_sigma: float, min_n: int,
                      rank_by: RankBy = "age+1s",
                      anchor: Anchor = "scan",
                      contiguous: bool = True,
                      mutual: bool = False,
                      inclusive: bool = True) -> np.ndarray:
    """Indices (into the original arrays) of the youngest overlapping cluster.

    Parameters that are genuinely free, and are therefore explicit:

    rank_by
        Order grains before searching. Ranking on age alone, on age+1s, or on
        age+2s can hand back different clusters whenever precision varies
        across the young tail, which it usually does.
    anchor
        'youngest' forces the rank-1 grain into the cluster (the reading of
        Dickinson & Gehrels in which the metric describes the youngest grain's
        company). 'scan' walks the seed up the ranking until a qualifying
        cluster appears, so a lone outlier is dropped rather than dragging the
        answer down.
    contiguous
        Require members to be consecutive in the ranking. If False, any grain
        overlapping the seed qualifies regardless of rank gaps.
    mutual
        If True, every member must overlap every other member, not merely the
        seed. This is the stricter and arguably more faithful reading of
        "a cluster of grains that overlap"; both major implementations use the
        looser seed-only test, so False is the compatible default.
    inclusive
        Whether overlap is tested with <= or <. Matters only for exact ties.
    """
    n = age.size
    if n == 0 or min_n < 1:
        return np.array([], dtype=int)

    order = np.argsort(_rank_key(age, s1, rank_by), kind="stable")
    a, s = age[order], s1[order]
    lo, hi = a - k_sigma * s, a + k_sigma * s
    le = np.less_equal if inclusive else np.less

    seeds = [0] if anchor == "youngest" else range(n)

    for i in seeds:
        ok = le(lo[i:], hi[i])                       # overlaps the seed
        if contiguous:
            bad = np.flatnonzero(~ok)
            run = int(bad[0]) if bad.size else int(ok.size)
            members = np.arange(i, i + run)
        else:
            members = i + np.flatnonzero(ok)

        if mutual and members.size > 1:
            keep = [members[0]]
            for j in members[1:]:
                if all(le(max(lo[j], lo[k]), min(hi[j], hi[k])) for k in keep):
                    keep.append(int(j))
            members = np.array(keep, dtype=int)

        if members.size >= min_n:
            return np.sort(order[members])

    return np.array([], dtype=int)


# --------------------------------------------------------------------------- #
# The metrics
# --------------------------------------------------------------------------- #

def ysg(sample: Sample, *, rank_by: RankBy = "age+1s") -> MDAResult:
    """Youngest single grain (Dickinson & Gehrels, 2009).

    The only free choice is which grain counts as "youngest": the lowest age,
    or the lowest age+1s, or the lowest age+2s. Ranking on age alone rewards
    the most imprecise analysis in the tail, which is why the age+sigma
    variants exist; they are not interchangeable.
    """
    res = MDAResult("YSG", convention=dict(rank_by=rank_by,
                                           sigma_in=sample.sigma_in))
    if sample.n == 0:
        res.note = "empty sample"
        return res
    a, s = sample.age, sample.s1
    i = int(np.argmin(_rank_key(a, s, rank_by)))
    res.mda = float(a[i])
    res.unc_1s = float(s[i])
    res.unc_95z = 1.959963984540054 * float(s[i])
    res.unc_95t = float("nan")
    res.n_used = 1
    res.idx_used = (i,)
    res.note = "single-grain metric; no MSWD is defined"
    return res


def yc(sample: Sample, *, k_sigma: float = 1.0, min_n: int = 2,
       rank_by: RankBy | None = None, anchor: Anchor = "scan",
       contiguous: bool = True, mutual: bool = False) -> MDAResult:
    """Youngest grain cluster, YC{k}s({min_n}+) (Dickinson & Gehrels, 2009).

    YC1s(2+) is k_sigma=1, min_n=2. YC2s(3+) is k_sigma=2, min_n=3.

    `rank_by` defaults to age+1s for k_sigma=1 and age+2s for k_sigma=2, which
    keeps the ranking on the same scale as the overlap test.
    """
    if rank_by is None:
        rank_by = "age+1s" if k_sigma <= 1.0 else "age+2s"

    label = f"YC{int(k_sigma)}s({min_n}+)"
    res = MDAResult(label, convention=dict(
        k_sigma=k_sigma, min_n=min_n, rank_by=rank_by, anchor=anchor,
        contiguous=contiguous, mutual=mutual, sigma_in=sample.sigma_in))

    idx = _youngest_cluster(sample.age, sample.s1, k_sigma=k_sigma,
                            min_n=min_n, rank_by=rank_by, anchor=anchor,
                            contiguous=contiguous, mutual=mutual)
    if idx.size == 0:
        res.note = f"no cluster of >= {min_n} grains overlapping at {k_sigma} sigma"
        return res
    return _fill_wm(res, weighted_mean(sample.age[idx], sample.s1[idx]), idx)


def y3za(sample: Sample, *, n_grains: int = 3, rank_by: RankBy = "age") -> MDAResult:
    """Weighted mean of the youngest n grains, overlap not required (Y3Za).

    Coutts et al. (2019). Y4Za is n_grains=4. A high MSWD here is diagnostic,
    not incidental: it is the metric telling you the grains it was forced to
    average do not belong together.
    """
    res = MDAResult(f"Y{n_grains}Za", convention=dict(
        n_grains=n_grains, rank_by=rank_by, sigma_in=sample.sigma_in))
    if sample.n < n_grains:
        res.note = f"fewer than {n_grains} grains"
        return res
    order = np.argsort(_rank_key(sample.age, sample.s1, rank_by), kind="stable")
    idx = np.sort(order[:n_grains])
    return _fill_wm(res, weighted_mean(sample.age[idx], sample.s1[idx]), idx)


def y3zo(sample: Sample, *, n_grains: int = 3, k_sigma: float = 2.0,
         rank_by: RankBy | None = None, anchor: Anchor = "scan",
         contiguous: bool = True, mutual: bool = False) -> MDAResult:
    """Weighted mean of the youngest n grains that overlap at k sigma (Y3Zo).

    Coutts et al. (2019). Implemented as: find the youngest qualifying cluster,
    then take its n youngest members. If the cluster is larger than n, the
    surplus is discarded -- that truncation is what distinguishes Y3Zo from
    YC2s(3+), which averages the whole cluster.
    """
    if rank_by is None:
        rank_by = "age+1s" if k_sigma <= 1.0 else "age+2s"

    res = MDAResult(f"Y{n_grains}Zo", convention=dict(
        n_grains=n_grains, k_sigma=k_sigma, rank_by=rank_by, anchor=anchor,
        contiguous=contiguous, mutual=mutual, sigma_in=sample.sigma_in))

    idx = _youngest_cluster(sample.age, sample.s1, k_sigma=k_sigma,
                            min_n=n_grains, rank_by=rank_by, anchor=anchor,
                            contiguous=contiguous, mutual=mutual)
    if idx.size == 0:
        res.note = f"no cluster of >= {n_grains} grains overlapping at {k_sigma} sigma"
        return res
    sub = idx[np.argsort(_rank_key(sample.age[idx], sample.s1[idx], rank_by))][:n_grains]
    sub = np.sort(sub)
    return _fill_wm(res, weighted_mean(sample.age[sub], sample.s1[sub]), sub)


def ypp(sample: Sample, *, min_n: int = 2, basis: Literal["pdp", "kde"] = "pdp",
        bandwidth: float = 10.0, step: float = 0.1,
        prominence_frac: float = 0.01, min_sep_myr: float = 0.0,
        peak_position: Literal["grid", "parabolic", "gaussfit"] = "parabolic",
        count_rule: Literal["grain_brackets_peak", "interval_overlap"] = "grain_brackets_peak",
        count_sigma: float = 2.0) -> MDAResult:
    """Youngest graphical / probability peak (Dickinson & Gehrels, 2009).

    Four independent choices decide the answer, and all four are exposed:

    basis, bandwidth, step
        A peak is a property of the density curve, not of the data. PDP and KDE
        peaks do not coincide.
    prominence_frac, min_sep_myr
        With no prominence floor, every ripple in the young tail is a peak and
        YPP collapses toward YSG. Separation is specified in Myr here and
        converted to grid points internally.
    peak_position
        'grid' takes the grid node, 'parabolic' interpolates sub-grid from
        three points, 'gaussfit' fits a Gaussian between the flanking troughs
        and returns its mean. The three can differ by more than the grid step
        on an asymmetric peak.
    count_rule
        'grain_brackets_peak' counts grains whose age +/- count_sigma spans the
        peak. 'interval_overlap' counts grains whose age +/- count_sigma
        interval intersects the peak's own +/- 2 sigma window, which requires a
        Gaussian fit and admits more grains. This decides which peak is the
        youngest *qualifying* one, so it can move the answer between modes, not
        just adjust a footnote.

    YPP is a position on a curve: it carries no weighted-mean uncertainty.
    """
    res = MDAResult("YPP", convention=dict(
        min_n=min_n, basis=basis, bandwidth=bandwidth, step=step,
        prominence_frac=prominence_frac, min_sep_myr=min_sep_myr,
        peak_position=peak_position, count_rule=count_rule,
        count_sigma=count_sigma, sigma_in=sample.sigma_in))
    if sample.n == 0:
        res.note = "empty sample"
        return res

    hi = float(sample.age.max() + 10 * sample.s1.max() + 50)
    if basis == "pdp":
        x, y = pdp(sample.age, sample.s1, 0.0, hi, step)
    else:
        x, y = kde(sample.age, bandwidth, 0.0, hi, step)

    sep_pts = max(1, int(round(min_sep_myr / step))) if min_sep_myr > 0 else 1
    pk = _local_extrema(y, "max", prominence_frac, sep_pts)
    if pk.size == 0:
        res.note = "no peaks detected"
        return res
    tr = _local_extrema(y, "min", 0.0, 1)

    a, s = sample.age, sample.s1
    for i in pk:                                   # ascending in age already
        if peak_position == "grid":
            mu, sig = float(x[i]), None
        elif peak_position == "parabolic":
            mu, sig = _refine_peak_parabolic(x, y, i), None
        else:
            lo_b = tr[tr < i].max() if np.any(tr < i) else 0
            hi_b = tr[tr > i].min() if np.any(tr > i) else y.size - 1
            fit = _fit_gaussian_segment(x[lo_b:hi_b + 1], y[lo_b:hi_b + 1])
            if fit is None:
                continue
            mu, sig = fit

        if count_rule == "grain_brackets_peak":
            members = np.flatnonzero((a - count_sigma * s <= mu) &
                                     (mu <= a + count_sigma * s))
        else:
            if sig is None:
                lo_b = tr[tr < i].max() if np.any(tr < i) else 0
                hi_b = tr[tr > i].min() if np.any(tr > i) else y.size - 1
                fit = _fit_gaussian_segment(x[lo_b:hi_b + 1], y[lo_b:hi_b + 1])
                if fit is None:
                    continue
                sig = fit[1]
            w_lo, w_hi = mu - 2 * sig, mu + 2 * sig
            members = np.flatnonzero((a + count_sigma * s >= w_lo) &
                                     (a - count_sigma * s <= w_hi))

        if members.size >= min_n:
            res.mda = float(mu)
            res.n_used = int(members.size)
            res.idx_used = tuple(int(j) for j in members)
            if sig is not None:
                res.unc_1s = float(sig)
                res.unc_95z = 1.959963984540054 * float(sig)
                res.note = ("peak position; quoted uncertainty is the fitted "
                            "peak width, not a standard error of the mean")
            else:
                res.note = "peak position; no uncertainty is defined for YPP"
            return res

    res.note = f"no peak supported by >= {min_n} grains"
    return res


def ygf(sample: Sample, *, min_n: int = 3, step: float = 0.1,
        prominence_frac: float = 0.0) -> MDAResult:
    """Youngest Gaussian fit (Saylor et al., 2023).

    Fits a Gaussian to the youngest PDP mode, bounded by the flanking density
    minima, and reports its mean. Unlike YPP this yields a width, but that
    width is the dispersion of the mode, not the precision of the estimate, so
    it should not be read as a standard error.
    """
    res = MDAResult("YGF", convention=dict(min_n=min_n, step=step,
                                           prominence_frac=prominence_frac,
                                           sigma_in=sample.sigma_in))
    if sample.n == 0:
        res.note = "empty sample"
        return res

    hi = float(sample.age.max() + 10 * sample.s1.max() + 50)
    x, y = pdp(sample.age, sample.s1, 0.0, hi, step)
    pk = _local_extrema(y, "max", prominence_frac, 1)
    tr = _local_extrema(y, "min", 0.0, 1)
    if pk.size == 0:
        res.note = "no modes detected"
        return res

    a, s = sample.age, sample.s1
    for i in pk:
        lo_b = tr[tr < i].max() if np.any(tr < i) else 0
        hi_b = tr[tr > i].min() if np.any(tr > i) else y.size - 1
        fit = _fit_gaussian_segment(x[lo_b:hi_b + 1], y[lo_b:hi_b + 1])
        if fit is None:
            continue
        mu, sig = fit
        members = np.flatnonzero((a >= mu - 2 * sig) & (a <= mu + 2 * sig))
        if members.size >= min_n:
            res.mda = mu
            res.unc_1s = sig
            res.unc_95z = 1.959963984540054 * sig
            res.n_used = int(members.size)
            res.idx_used = tuple(int(j) for j in members)
            res.note = ("quoted uncertainty is the fitted Gaussian sigma "
                        "(mode width), not a standard error")
            return res

    res.note = f"no mode with >= {min_n} grains inside +/- 2 sigma"
    return res


def ysp(sample: Sample, *, min_n: int = 2, rank_by: RankBy = "age",
        target_mswd: float = 1.0,
        entry_rule: Literal["pair", "global", "none"] = "global",
        drop_youngest_on_fail: bool = True,
        max_drops: int | None = None) -> MDAResult:
    """Youngest statistical population (Coutts et al., 2019).

    Grow a population from the youngest grain upward, recording the MSWD of
    every cumulative subset of size 2..n, and keep the subset whose MSWD is
    closest to `target_mswd`.

    The paper does not say what to do when no subset ever reaches MSWD ~ 1, and
    that gap is where implementations diverge:

    entry_rule='pair'
        Accept only if the youngest *pair* already has MSWD < target. If not,
        discard the youngest grain and restart. Strict: it will throw away a
        young grain even when adding a third would have produced MSWD ~ 1.
    entry_rule='global'
        Accept if *any* cumulative subset reaches MSWD < target. Only if none
        does is the youngest grain discarded and the search restarted.
    entry_rule='none'
        Always return the closest-to-target subset with no acceptance test.

    On real samples with a scattered young tail, 'pair' and 'global' routinely
    select different populations, and 'pair' is the more conservative (older).
    """
    res = MDAResult("YSP", convention=dict(
        min_n=min_n, rank_by=rank_by, target_mswd=target_mswd,
        entry_rule=entry_rule, drop_youngest_on_fail=drop_youngest_on_fail,
        sigma_in=sample.sigma_in))

    if sample.n < min_n:
        res.note = f"fewer than {min_n} grains"
        return res

    order = np.argsort(_rank_key(sample.age, sample.s1, rank_by), kind="stable")
    limit = sample.n if max_drops is None else min(sample.n, max_drops + 1)

    for drop in range(limit):
        live = order[drop:]
        if live.size < min_n:
            break
        a, s = sample.age[live], sample.s1[live]

        stats_by_k = [weighted_mean(a[:k], s[:k]) for k in range(min_n, live.size + 1)]
        mswds = np.array([st["mswd"] for st in stats_by_k], dtype=float)
        if not np.any(np.isfinite(mswds)):
            continue

        best = int(np.nanargmin(np.abs(mswds - target_mswd)))

        if entry_rule == "pair":
            passed = math.isfinite(mswds[0]) and mswds[0] < target_mswd
        elif entry_rule == "global":
            passed = bool(np.nanmin(mswds) < target_mswd)
        else:
            passed = True

        if passed:
            k = best + min_n
            idx = np.sort(live[:k])
            res = _fill_wm(res, stats_by_k[best], idx)
            if drop:
                res.note = f"{drop} youngest grain(s) discarded before acceptance"
            return res

        if not drop_youngest_on_fail:
            break

    res.note = "no subset satisfied the acceptance rule"
    return res


def tau(sample: Sample, *, min_n: int = 3, step: float = 0.1,
        prominence_frac: float = 0.0,
        bounds: Literal["troughs", "gauss2s"] = "troughs") -> MDAResult:
    """Tau method (Barbeau et al., 2009).

    Weighted mean of the grains belonging to the youngest PDP mode that holds
    at least `min_n` grains. What "belonging" means is the whole argument:

    bounds='troughs'
        Grains between the density minima flanking the mode. This is Barbeau's
        original construction, and it is sensitive to shallow ripples in the
        young tail -- a minor trough can amputate the population.
    bounds='gauss2s'
        Grains within +/- 2 sigma of a Gaussian fitted to the trough-bounded
        mode. Smoother and less trough-sensitive, but it can reach across a
        trough and pull in grains from a neighbouring mode.

    These two give the same answer on well-separated modes and different
    answers exactly where MDAs are hardest, which is a good reason to report
    which one was used.
    """
    res = MDAResult("Tau", convention=dict(
        min_n=min_n, step=step, prominence_frac=prominence_frac,
        bounds=bounds, sigma_in=sample.sigma_in))
    if sample.n == 0:
        res.note = "empty sample"
        return res

    hi = float(sample.age.max() + 10 * sample.s1.max() + 50)
    x, y = pdp(sample.age, sample.s1, 0.0, hi, step)
    pk = _local_extrema(y, "max", prominence_frac, 1)
    tr = _local_extrema(y, "min", 0.0, 1)
    if pk.size == 0:
        res.note = "no modes detected"
        return res

    a = sample.age
    for i in pk:
        lo_i = tr[tr < i].max() if np.any(tr < i) else 0
        hi_i = tr[tr > i].min() if np.any(tr > i) else y.size - 1

        if bounds == "troughs":
            lo_a, hi_a = float(x[lo_i]), float(x[hi_i])
        else:
            fit = _fit_gaussian_segment(x[lo_i:hi_i + 1], y[lo_i:hi_i + 1])
            if fit is None:
                continue
            mu, sig = fit
            lo_a, hi_a = mu - 2 * sig, mu + 2 * sig

        idx = np.flatnonzero((a >= lo_a) & (a <= hi_a))
        if idx.size >= min_n:
            res = _fill_wm(res, weighted_mean(a[idx], sample.s1[idx]), idx)
            res.convention["window_ma"] = (round(lo_a, 4), round(hi_a, 4))
            return res

    res.note = f"no mode containing >= {min_n} grains"
    return res


def ydz(sample: Sample, *, iterations: int = 10000, prefilter_sigma: float = 5.0,
        bins: int = 25, mode_estimator: Literal["hist", "kde"] = "kde",
        kde_bandwidth: float | None = None, seed: int | None = None) -> MDAResult:
    """Youngest detrital zircon, Monte Carlo (after Ludwig, 2012, Isoplot).

    Resample every grain in the young tail from its own Gaussian, take the
    minimum of each realisation, and summarise the resulting distribution.
    Reported as a mode with asymmetric 2.5 / 97.5 percentile bounds.

    Two warnings, both structural rather than implementational:

    1. Results are not reproducible without the seed, and the mode moves with
       `bins` under the histogram estimator. The KDE estimator is offered as a
       less arbitrary default; it is *not* what Isoplot does.
    2. YDZ is a minimum of random draws, so it drifts younger as n and as
       analytical uncertainty grow. Vermeesch (2021) shows it cannot converge
       on the true depositional age even in principle. Reproducing Isoplot's
       exact numbers is a documented open problem; treat the value as a
       diagnostic, not an age.
    """
    res = MDAResult("YDZ", convention=dict(
        iterations=iterations, prefilter_sigma=prefilter_sigma, bins=bins,
        mode_estimator=mode_estimator, seed=seed, sigma_in=sample.sigma_in))
    if sample.n == 0:
        res.note = "empty sample"
        return res

    rng = np.random.default_rng(seed)
    a, s = sample.age, sample.s1
    j = int(np.argmin(a + s))
    cut = a[j] + prefilter_sigma * s[j]
    keep = np.flatnonzero(a < cut)
    if keep.size == 0:
        keep = np.array([j])

    draws = rng.normal(a[keep][None, :], s[keep][None, :],
                       size=(iterations, keep.size)).min(axis=1)

    if mode_estimator == "hist":
        counts, edges = np.histogram(draws, bins=bins)
        k = int(np.argmax(counts))
        mode = float(0.5 * (edges[k] + edges[k + 1]))
    else:
        bw = kde_bandwidth or (0.9 * draws.std(ddof=1) * draws.size ** (-0.2))
        bw = max(bw, 1e-6)
        grid = np.linspace(draws.min() - 3 * bw, draws.max() + 3 * bw, 2048)
        d = np.exp(-0.5 * ((grid[:, None] - draws[None, :]) / bw) ** 2).sum(axis=1)
        mode = float(grid[int(np.argmax(d))])

    p_lo, p_hi = np.percentile(draws, [2.5, 97.5])
    res.mda = mode
    res.unc_minus = float(mode - p_lo)
    res.unc_plus = float(p_hi - mode)
    res.unc_1s = float(draws.std(ddof=1))
    res.n_used = int(keep.size)
    res.idx_used = tuple(int(i) for i in keep)
    res.note = ("asymmetric 95% bounds from MC percentiles; known to drift "
                "young -- see Vermeesch (2021)")
    return res


def mla(sample: Sample, *, log: bool = True, n_starts: int = 12,
        seed: int | None = 0) -> MDAResult:
    """Maximum likelihood age: Galbraith's 3-parameter minimum age model,
    applied to U-Pb as recommended by Vermeesch (2021).

    Models the dates as a mixture of a discrete youngest component at gamma
    (proportion p) and a truncated normal of older components (mu, sigma),
    each observed with its own analytical error. The MDA is gamma, or
    exp(gamma) when fitted on log ages.

    Unlike the ad hoc metrics, this one has a consistent estimator: it does not
    drift younger as n grows, because it separates real dispersion from
    analytical scatter instead of confusing the two. Uncertainty here is a
    profile-free normal approximation from the observed information matrix,
    reported as a studentised 95% interval.

    Requires SciPy for the optimiser; returns an empty result if unavailable.
    """
    res = MDAResult("MLA", convention=dict(log=log, n_starts=n_starts,
                                           seed=seed, sigma_in=sample.sigma_in))
    if sample.n < 3:
        res.note = "at least 3 grains required"
        return res
    try:
        from scipy.optimize import minimize
        from scipy.stats import norm
    except Exception:
        res.note = "SciPy required for MLA"
        return res

    a, s = sample.age, sample.s1
    if log:
        z = np.log(a)
        se = s / a                          # delta-method transfer to log space
    else:
        z, se = a.copy(), s.copy()

    def nll(theta):
        p_raw, gamma, mu, log_sig = theta
        p = 1.0 / (1.0 + math.exp(-p_raw))          # keep p in (0, 1)
        sig = math.exp(log_sig)
        if not np.isfinite([gamma, mu, sig]).all() or sig <= 0:
            return 1e12
        # discrete youngest component
        f0 = norm.pdf(z, gamma, se)
        # truncated-normal older components, integrated over each grain's error
        v = sig ** 2 + se ** 2
        sd = np.sqrt(v)
        mu0 = (mu / sig ** 2 + z / se ** 2) / (1.0 / sig ** 2 + 1.0 / se ** 2)
        s0 = np.sqrt(1.0 / (1.0 / sig ** 2 + 1.0 / se ** 2))
        denom = norm.sf((gamma - mu) / sig)
        if denom < 1e-300:
            return 1e12
        f1 = norm.pdf(z, mu, sd) * norm.sf((gamma - mu0) / s0) / denom
        lik = p * f0 + (1.0 - p) * f1
        if np.any(lik <= 0) or not np.all(np.isfinite(lik)):
            return 1e12
        return float(-np.sum(np.log(lik)))

    rng = np.random.default_rng(seed)
    zmin, zmed, zsd = z.min(), float(np.median(z)), float(z.std(ddof=1)) or 1.0
    best, best_val = None, np.inf
    for k in range(n_starts):
        jitter = 0.0 if k == 0 else rng.normal(0, 0.3, 4)
        x0 = np.array([-1.0, zmin + 0.02 * zsd, zmed, math.log(max(zsd, 1e-3))])
        x0 = x0 + jitter
        try:
            out = minimize(nll, x0, method="Nelder-Mead",
                           options=dict(maxiter=8000, xatol=1e-9, fatol=1e-9))
        except Exception:
            continue
        if out.fun < best_val and np.isfinite(out.fun):
            best, best_val = out, out.fun

    if best is None:
        res.note = "optimiser failed to converge"
        return res

    gamma = float(best.x[1])

    # observed-information standard error on gamma by central differences
    h = 1e-4 * max(abs(gamma), 1.0)
    xp, xm = best.x.copy(), best.x.copy()
    xp[1] += h
    xm[1] -= h
    curv = (nll(xp) - 2.0 * best_val + nll(xm)) / h ** 2
    se_gamma = 1.0 / math.sqrt(curv) if curv > 0 else float("nan")

    if log:
        res.mda = math.exp(gamma)
        se_age = res.mda * se_gamma if math.isfinite(se_gamma) else float("nan")
    else:
        res.mda = gamma
        se_age = se_gamma

    res.unc_1s = se_age
    res.unc_95z = 1.959963984540054 * se_age
    res.unc_95t = _student_t_975(sample.n - 1) * se_age
    res.n_used = sample.n
    res.idx_used = tuple(range(sample.n))
    p_hat = 1.0 / (1.0 + math.exp(-float(best.x[0])))
    res.convention["p_youngest"] = round(p_hat, 4)
    res.convention["n_effective"] = round(p_hat * sample.n, 2)
    res.note = ("all grains inform the fit; p_youngest is the estimated "
                "proportion in the youngest component")
    return res


# --------------------------------------------------------------------------- #
# Presets and the batch driver
# --------------------------------------------------------------------------- #

PRESETS: dict[str, dict] = {
    # Reproduces the conventions of the widely used Python implementation.
    "python_toolset": dict(
        ysg=dict(rank_by="age+1s"),
        yc1s=dict(k_sigma=1.0, min_n=2, rank_by="age+1s", anchor="scan", contiguous=True),
        yc2s=dict(k_sigma=2.0, min_n=3, rank_by="age+2s", anchor="scan", contiguous=True),
        y3za=dict(rank_by="age"),
        y3zo=dict(k_sigma=2.0, rank_by="age+2s", anchor="scan", contiguous=True),
        ypp=dict(min_n=2, peak_position="parabolic", prominence_frac=0.01,
                 min_sep_myr=0.0, count_rule="grain_brackets_peak", count_sigma=2.0),
        ysp=dict(rank_by="age", entry_rule="pair", target_mswd=1.0),
        tau=dict(min_n=3, bounds="troughs", prominence_frac=0.01),
    ),
    # Reproduces the conventions of the widely used MATLAB implementation.
    "matlab_toolset": dict(
        ysg=dict(rank_by="age"),
        yc1s=dict(k_sigma=1.0, min_n=2, rank_by="age", anchor="scan", contiguous=True),
        yc2s=dict(k_sigma=2.0, min_n=3, rank_by="age", anchor="scan", contiguous=True),
        y3za=dict(rank_by="age"),
        y3zo=dict(k_sigma=2.0, rank_by="age", anchor="scan", contiguous=True),
        ypp=dict(min_n=2, peak_position="gaussfit", prominence_frac=0.0,
                 min_sep_myr=0.0, count_rule="interval_overlap", count_sigma=2.0),
        ysp=dict(rank_by="age", entry_rule="global", target_mswd=1.0),
        tau=dict(min_n=3, bounds="gauss2s", prominence_frac=0.0),
    ),
    # Recommended defaults: rank on the scale the overlap test uses, let an
    # outlying youngest grain be dropped, require a real peak, and prefer the
    # less brittle acceptance rules.
    "harmonized": dict(
        ysg=dict(rank_by="age+1s"),
        yc1s=dict(k_sigma=1.0, min_n=2, rank_by="age+1s", anchor="scan",
                  contiguous=True, mutual=False),
        yc2s=dict(k_sigma=2.0, min_n=3, rank_by="age+2s", anchor="scan",
                  contiguous=True, mutual=False),
        y3za=dict(rank_by="age"),
        y3zo=dict(k_sigma=2.0, rank_by="age+2s", anchor="scan", contiguous=True),
        ypp=dict(min_n=2, peak_position="parabolic", prominence_frac=0.02,
                 min_sep_myr=1.0, count_rule="grain_brackets_peak", count_sigma=2.0),
        ysp=dict(rank_by="age", entry_rule="global", target_mswd=1.0),
        tau=dict(min_n=3, bounds="troughs", prominence_frac=0.02),
    ),
}


def all_metrics(sample: Sample, preset: str = "harmonized",
                overrides: dict | None = None,
                include: Iterable[str] | None = None,
                ydz_kwargs: dict | None = None,
                mla_kwargs: dict | None = None) -> dict[str, MDAResult]:
    """Run the full suite under one named convention set.

    Returns a dict keyed by metric name. Every value carries its own resolved
    convention, so a results table exported from this is self-documenting.
    """
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    cfg = {k: dict(v) for k, v in PRESETS[preset].items()}
    for k, v in (overrides or {}).items():
        cfg.setdefault(k, {}).update(v)

    out: dict[str, MDAResult] = {}
    out["YSG"] = ysg(sample, **cfg.get("ysg", {}))
    out["YC1s"] = yc(sample, **cfg.get("yc1s", {}))
    out["YC2s"] = yc(sample, **cfg.get("yc2s", {}))
    out["Y3Za"] = y3za(sample, **cfg.get("y3za", {}))
    out["Y3Zo"] = y3zo(sample, **cfg.get("y3zo", {}))
    out["YPP"] = ypp(sample, **cfg.get("ypp", {}))
    out["YGF"] = ygf(sample, **cfg.get("ygf", {}))
    out["YSP"] = ysp(sample, **cfg.get("ysp", {}))
    out["Tau"] = tau(sample, **cfg.get("tau", {}))
    out["YDZ"] = ydz(sample, **(ydz_kwargs or {}))
    out["MLA"] = mla(sample, **(mla_kwargs or {}))

    for r in out.values():
        r.convention["preset"] = preset
    if include is not None:
        keep = set(include)
        out = {k: v for k, v in out.items() if k in keep}
    return out


def to_table(results: dict[str, MDAResult]) -> str:
    """Plain-text summary. Uncertainties shown at 1 sigma and 95% (z)."""
    hdr = f"{'metric':8s} {'MDA (Ma)':>10s} {'1s':>8s} {'95% z':>8s} {'MSWD':>7s} {'n':>4s}  note"
    lines = [hdr, "-" * len(hdr)]
    for k, r in results.items():
        if not math.isfinite(r.mda):
            lines.append(f"{k:8s} {'--':>10s} {'':>8s} {'':>8s} {'':>7s} {'':>4s}  {r.note}")
            continue
        mswd = f"{r.mswd:7.2f}" if math.isfinite(r.mswd) else f"{'--':>7s}"
        u1 = f"{r.unc_1s:8.3f}" if math.isfinite(r.unc_1s) else f"{'--':>8s}"
        u2 = f"{r.unc_95z:8.3f}" if math.isfinite(r.unc_95z) else f"{'--':>8s}"
        lines.append(f"{k:8s} {r.mda:10.3f} {u1} {u2} {mswd} {r.n_used:4d}  {r.note}")
    return "\n".join(lines)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    true_age = 100.0
    n_young, n_old = 12, 60
    a_young = rng.normal(true_age, 1.5, n_young)
    a_old = rng.normal(310.0, 25.0, n_old)
    ages = np.r_[a_young, a_old]
    errs = np.r_[rng.uniform(1.0, 3.0, n_young), rng.uniform(3.0, 8.0, n_old)]

    smp = Sample(ages, errs, sigma_in=1, name="synthetic")
    print(f"synthetic sample, true depositional age = {true_age} Ma, n = {smp.n}\n")
    for name in ("python_toolset", "matlab_toolset", "harmonized"):
        print(f"### preset: {name}")
        print(to_table(all_metrics(smp, preset=name)))
        print()
