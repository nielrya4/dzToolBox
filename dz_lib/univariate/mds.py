from dz_lib.univariate import distributions, metrics
from dz_lib.univariate.data import Sample
from dz_lib.utils import fonts
from sklearn.manifold import MDS
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class MDSPoint:
    def __init__(self, x: float, y: float, label: str, nearest_neighbor: tuple = None):
        self.x = x
        self.y = y
        self.label = label
        self.nearest_neighbor = nearest_neighbor  # (x, y) or None


def _compute_dissimilarity_matrix(samples: list[Sample], metric: str = "similarity"):
    """
    Build the pairwise dissimilarity matrix for a list of samples.

    Similarity, likeness and cross-correlation compare the samples' PDPs (KDEs when the
    caller has set grain uncertainties to a bandwidth). KS D and Kuiper V compare the
    empirical CDFs of the ages, as in DZmds and detritalPy.

    Returns
    -------
    dissimilarity_matrix : ndarray, shape (n, n)
    prob_distros : list of distribution objects
    """
    n_samples = len(samples)
    dissimilarity_matrix = np.zeros((n_samples, n_samples))
    prob_distros = [distributions.pdp_function(sample) for sample in samples]
    ages = [sample.get_ages() for sample in samples]

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if metric == "similarity":
                value = metrics.dis_similarity(prob_distros[i].y_values, prob_distros[j].y_values)
            elif metric == "likeness":
                value = metrics.dis_likeness(prob_distros[i].y_values, prob_distros[j].y_values)
            elif metric == "cross_correlation":
                value = metrics.dis_r2(prob_distros[i].y_values, prob_distros[j].y_values)
            elif metric == "ks":
                value = metrics.ks_ages(ages[i], ages[j])
            elif metric == "kuiper":
                value = metrics.kuiper_ages(ages[i], ages[j])
            else:
                raise ValueError(f"Unknown metric '{metric}'")

            dissimilarity_matrix[i, j] = value
            dissimilarity_matrix[j, i] = value  # matrix is symmetric

    return dissimilarity_matrix, prob_distros


def _classical_mds(dissimilarity_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Classical (Torgerson) MDS, MATLAB cmdscale: the default start for mdscale."""
    n = dissimilarity_matrix.shape[0]
    centering = np.eye(n) - 1.0 / n
    b = -0.5 * centering @ (dissimilarity_matrix ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(b)
    top = np.argsort(eigenvalues)[::-1][:n_components]
    return eigenvectors[:, top] * np.sqrt(np.maximum(eigenvalues[top], 1e-12))


def _kruskal_stress(dissimilarity_matrix: np.ndarray, embedding: np.ndarray, non_metric: bool):
    """Kruskal stress-1 of `embedding`, plus its pairwise distances and fitted disparities."""
    upper = np.triu_indices_from(dissimilarity_matrix, k=1)
    dissimilarities = dissimilarity_matrix[upper]
    distances = np.linalg.norm(embedding[upper[0]] - embedding[upper[1]], axis=1)
    if non_metric:
        fitted = IsotonicRegression().fit_transform(dissimilarities, distances)
    else:
        fitted = dissimilarities
    denom = np.sum(distances ** 2)
    stress = float(np.sqrt(np.sum((distances - fitted) ** 2) / denom)) if denom > 0 else 0.0
    return stress, dissimilarities, distances


def _polish_stress1(dissimilarity_matrix: np.ndarray, embedding: np.ndarray, non_metric: bool) -> np.ndarray:
    """Minimise Kruskal stress-1 directly from a good start (mdscale's 'stress' criterion).

    SMACOF minimises raw stress, which is close to but not the same as stress-1, so a few
    L-BFGS steps on stress-1 itself finish the job. The result is kept only if it lowers
    stress-1 without collapsing the map (non-metric stress-1 is also zero when every
    distance is equal, or when points merge into tight clusters).
    """
    n, dims = embedding.shape
    upper = np.triu_indices(n, k=1)
    dissimilarities = dissimilarity_matrix[upper]
    isotonic = IsotonicRegression()

    def stress_and_gradient(flat):
        coords = flat.reshape(n, dims)
        diff = coords[:, None, :] - coords[None, :, :]
        dist_full = np.sqrt((diff ** 2).sum(-1))
        d = dist_full[upper]
        d_hat = isotonic.fit_transform(dissimilarities, d) if non_metric else dissimilarities
        raw, norm = np.sum((d - d_hat) ** 2), np.sum(d ** 2)
        stress = np.sqrt(raw / norm)
        grad_d = ((d - d_hat) / norm - raw * d / norm ** 2) / max(stress, 1e-15)
        g = np.zeros((n, n))
        g[upper] = grad_d
        g = g + g.T
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(dist_full > 0, g / dist_full, 0.0)
        return stress, (weights[:, :, None] * diff).sum(1).ravel()

    start_stress = stress_and_gradient(embedding.ravel())[0]
    result = minimize(stress_and_gradient, embedding.ravel(), jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000})
    polished = result.x.reshape(n, dims)
    d = np.linalg.norm(polished[upper[0]] - polished[upper[1]], axis=1)
    collapsed = d.mean() == 0 or d.std() / d.mean() < 0.1 or np.mean(d < 1e-3 * d.max()) > 0.05
    if result.fun < start_stress and not collapsed:
        return polished - polished.mean(0)
    return embedding


def _compute_mds(dissimilarity_matrix: np.ndarray, non_metric: bool = True):
    """
    Run MDS on a precomputed dissimilarity matrix.

    - Metric MDS: SMACOF from 10 seeded random starts plus the classical MDS solution.
    - Non-metric MDS: SMACOF started from the classical MDS solution and from the metric
      MDS solution; the start reaching the lower stress-1 wins. Random starts are not used:
      with more than ~12 samples, scikit-learn's non-metric SMACOF from random starts
      stalls far from the optimum (stress-1 ~0.4 where ~0.05 is reachable), giving maps
      that bear no relation to the data and change from run to run.
    - The winner is then polished by minimising stress-1 directly (_polish_stress1).
    - Stress is Kruskal stress-1, sqrt(sum((d - d_hat)^2) / sum(d^2)), with d_hat the
      dissimilarities (metric) or their isotonic fit to d (non-metric). sklearn's raw stress
      can't be used directly: for non-metric MDS it is on sklearn's internal scale.
    - Non-metric embeddings come out on an arbitrary scale, so they are rescaled so the
      largest distance equals the largest dissimilarity, the convention of MATLAB's mdscale
      (and so DZmds; Saylor et al., 2017). Shape and stress are unchanged by this.

    Returns
    -------
    mds_result   : fitted sklearn MDS object of the chosen solution
    embedding    : ndarray, shape (n, 2), 2-D coordinates in dissimilarity units
    kruskal_stress : float, Kruskal stress-1 in [0, 1]
    """
    classical = _classical_mds(dissimilarity_matrix)

    def smacof(metric, init=None, n_init=1, random_state=None):
        model = MDS(n_components=2, dissimilarity="precomputed", metric=metric,
                    normalized_stress=False, n_init=n_init, max_iter=1000, random_state=random_state)
        return model, model.fit_transform(dissimilarity_matrix, init=init)

    metric_candidates = [smacof(True, n_init=10, random_state=0), smacof(True, init=classical)]
    if non_metric:
        best_metric = min(metric_candidates,
                          key=lambda c: _kruskal_stress(dissimilarity_matrix, c[1], False)[0])
        candidates = [smacof(False, init=classical), smacof(False, init=best_metric[1])]
    else:
        candidates = metric_candidates
    mds_result, embedding = min(
        candidates, key=lambda c: _kruskal_stress(dissimilarity_matrix, c[1], non_metric)[0])
    embedding = _polish_stress1(dissimilarity_matrix, embedding, non_metric)

    _, dissimilarities, distances = _kruskal_stress(dissimilarity_matrix, embedding, non_metric)
    if non_metric and distances.max() > 0:
        embedding = embedding * (dissimilarities.max() / distances.max())
    kruskal_stress, _, _ = _kruskal_stress(dissimilarity_matrix, embedding, non_metric)

    return mds_result, embedding, kruskal_stress


def mds_function(samples: list[Sample], metric: str = "similarity", non_metric: bool = True):
    """
    Compute MDS for a list of samples.

    Returns
    -------
    points             : list of MDSPoint
    kruskal_stress     : float, Kruskal stress-1
    dissimilarity_matrix : ndarray
    embedding          : ndarray, shape (n, 2)
    mds_result         : fitted sklearn MDS object
    """
    n_samples = len(samples)
    dissimilarity_matrix, prob_distros = _compute_dissimilarity_matrix(samples, metric)
    mds_result, embedding, kruskal_stress = _compute_mds(dissimilarity_matrix, non_metric=non_metric)

    # Build an index map once so nearest-neighbour lookup is O(1)
    sample_index = {id(s): i for i, s in enumerate(samples)}

    points = []
    for i in range(n_samples):
        # Find nearest neighbour (lowest dissimilarity, excluding self)
        row = dissimilarity_matrix[i].copy()
        row[i] = np.inf
        j = int(np.argmin(row))

        x1, y1 = embedding[i]
        x2, y2 = embedding[j]
        points.append(MDSPoint(x1, y1, samples[i].name, nearest_neighbor=(x2, y2)))

    return points, kruskal_stress, dissimilarity_matrix, embedding, mds_result


def mds_graph(
        points: list[MDSPoint],
        title: str = None,
        font_path: str = None,
        font_size: float = 12,
        fig_width: float = 9,
        fig_height: float = 7,
        color_map: str = "plasma",
):
    """
    Plot 2-D MDS coordinates.

    Fixes vs. original
    ------------------
    - Nearest-neighbour line guard now checks `point.nearest_neighbor is not None`
      (the original `if (x2, y2) is not None` was always True).
    - Uses `matplotlib.colormaps` instead of the deprecated `plt.cm.get_cmap`.
    """
    n_samples = len(points)
    cmap = matplotlib.colormaps[color_map]
    colors = cmap(np.linspace(0, 1, n_samples))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    for i, point in enumerate(points):
        ax.scatter(point.x, point.y, color=colors[i])
        ax.text(point.x, point.y + 0.005, point.label,
                fontsize=font_size * 0.75, ha="center", va="center")

        # Fixed guard: tuples are never None, so we must check the attribute
        if point.nearest_neighbor is not None:
            x2, y2 = point.nearest_neighbor
            ax.plot([point.x, x2], [point.y, y2], "k--", linewidth=0.5)

    font = fonts.get_font(font_path) if font_path else fonts.get_default_font()
    fig.suptitle(title, fontsize=font_size * 1.75, fontproperties=font)
    fig.text(0.5, 0.01, "Dimension 1", ha="center", va="center",
             fontsize=font_size, fontproperties=font)
    fig.text(0.01, 0.5, "Dimension 2", va="center", rotation="vertical",
             fontsize=font_size, fontproperties=font)

    fig.tight_layout()
    plt.close()
    return fig


def shepard_plot(
        dissimilarity_matrix: np.ndarray,
        embedding: np.ndarray,
        mds_result,
        kruskal_stress: float,
        non_metric: bool = True,
        title: str = "Shepard Plot",
        font_path: str = None,
        font_size: float = 12,
        fig_width: float = 8,
        fig_height: float = 6,
):
    """
    Shepard plot: original dissimilarities vs. MDS distances (and disparities).

    Fixes vs. original
    ------------------
    - Displays Kruskal stress-1 instead of raw sklearn stress.
    - For non-metric MDS the R² is now computed as the correlation between
      original dissimilarities and MDS distances (rank relationship), not
      between isotonic-fitted disparities and distances (which was inflated
      because the disparities are directly fitted to the distances).
    - x-axis limit is derived from the data instead of being hardcoded to
      [0, 1.02], which only holds if all metrics are guaranteed in [0, 1].
    """
    n_samples = dissimilarity_matrix.shape[0]
    upper = np.triu_indices(n_samples, k=1)

    original_dissimilarities = dissimilarity_matrix[upper].tolist()
    mds_distances = [
        float(np.linalg.norm(embedding[i] - embedding[j]))
        for i, j in zip(*upper)
    ]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    if non_metric:
        ax.scatter(original_dissimilarities, mds_distances,
                   alpha=0.7, s=30, facecolors="none", edgecolors="blue",
                   linewidth=1, label="Distances")

        # Isotonic (monotone) regression → disparities
        iso = IsotonicRegression()
        disparities = iso.fit_transform(original_dissimilarities, mds_distances)

        sorted_idx = np.argsort(original_dissimilarities)
        ax.plot(np.array(original_dissimilarities)[sorted_idx],
                disparities[sorted_idx],
                "r-", linewidth=2, alpha=0.8, label="Disparities")

        max_val = max(max(original_dissimilarities), max(mds_distances))
        ax.plot([0, max_val], [0, max_val], ":", color="black", alpha=0.6, label="1:1")

        ax.set_xlabel("Dissimilarities", fontsize=font_size)
        ax.set_ylabel("Distances / Disparities", fontsize=font_size)

        # R²: rank relationship between dissimilarities and MDS distances.
        # (Previously measured disparities vs distances, which was inflated
        # because disparities are isotonically fitted to match the distances.)
        r2 = float(np.corrcoef(original_dissimilarities, mds_distances)[0, 1] ** 2)

    else:
        ax.scatter(original_dissimilarities, mds_distances,
                   alpha=0.7, s=30, facecolors="none", edgecolors="blue",
                   linewidth=1, label="Distances")
        # Single 1:1 reference line from origin
        max_val = max(max(original_dissimilarities), max(mds_distances))
        ax.plot([0, max_val], [0, max_val], "r-", linewidth=2, alpha=0.8, label="1:1 (perfect fit)")
        ax.set_xlabel("Dissimilarities", fontsize=font_size)
        ax.set_ylabel("Distances", fontsize=font_size)
        r2 = float(np.corrcoef(original_dissimilarities, mds_distances)[0, 1] ** 2)

    ax.legend(loc="upper left")

    # Use Kruskal stress-1 (passed in, computed in _compute_mds)
    textstr = f"R² = {r2:.3f}\nStress-1 = {kruskal_stress:.5f}"
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
            fontsize=font_size * 0.9, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    font = fonts.get_font(font_path) if font_path else fonts.get_default_font()
    fig.suptitle(title, fontsize=font_size * 1.75, fontproperties=font)

    ax.grid(True, alpha=0.3)

    # x-axis derived from data, not hardcoded to [0, 1.02]
    x_max = max(original_dissimilarities) * 1.05
    plt.xlim([0, x_max])

    fig.tight_layout()
    plt.close()
    return fig