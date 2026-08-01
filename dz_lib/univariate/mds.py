from dz_lib.univariate import distributions, metrics
from dz_lib.univariate.data import Sample
from dz_lib.utils import fonts
from sklearn.manifold import MDS
from sklearn.isotonic import IsotonicRegression
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

    Returns
    -------
    dissimilarity_matrix : ndarray, shape (n, n)
    prob_distros : list of distribution objects
    c_distros : list of CDF distribution objects
    """
    n_samples = len(samples)
    dissimilarity_matrix = np.zeros((n_samples, n_samples))
    prob_distros = [distributions.pdp_function(sample) for sample in samples]
    c_distros = [distributions.cdf_function(prob_distro) for prob_distro in prob_distros]

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if metric == "similarity":
                value = metrics.dis_similarity(prob_distros[i].y_values, prob_distros[j].y_values)
            elif metric == "likeness":
                value = metrics.dis_likeness(prob_distros[i].y_values, prob_distros[j].y_values)
            elif metric == "cross_correlation":
                value = metrics.dis_r2(prob_distros[i].y_values, prob_distros[j].y_values)
            elif metric == "ks":
                value = metrics.ks(c_distros[i].y_values, c_distros[j].y_values)
            elif metric == "kuiper":
                value = metrics.kuiper(c_distros[i].y_values, c_distros[j].y_values)
            else:
                raise ValueError(f"Unknown metric '{metric}'")

            dissimilarity_matrix[i, j] = value
            dissimilarity_matrix[j, i] = value  # matrix is symmetric

    return dissimilarity_matrix, prob_distros, c_distros


def _compute_mds(dissimilarity_matrix: np.ndarray, non_metric: bool = True):
    """
    Run MDS on a precomputed dissimilarity matrix.

    Fixes vs. original
    ------------------
    - Exposes n_init and max_iter for a more stable solution.
    - Computes Kruskal stress-1 (normalized) instead of raw sklearn stress.
    - Re-scales the 2-D embedding so axis spread grows with mean dissimilarity,
      making high-dissimilarity datasets visually distinct from low-dissimilarity ones.

    Returns
    -------
    mds_result   : fitted sklearn MDS object (stress_ attribute is raw sklearn stress)
    embedding    : ndarray, shape (n, 2), re-scaled 2-D coordinates
    kruskal_stress : float, Kruskal stress-1 in [0, 1]
    """
    mds_result = MDS(
        n_components=2,
        dissimilarity="precomputed",
        metric=(not non_metric),
        normalized_stress=False,   # keep raw stress so we can normalize ourselves
        n_init=10,                 # more random restarts → more stable solution
        max_iter=1000,
    )
    embedding = mds_result.fit_transform(dissimilarity_matrix)

    # --- Kruskal stress-1 (normalized) ---
    upper = np.triu_indices_from(dissimilarity_matrix, k=1)
    denom = np.sum(dissimilarity_matrix[upper] ** 2)
    kruskal_stress = float(np.sqrt(mds_result.stress_ / denom)) if denom > 0 else 0.0

    # --- Re-scale so axis spread reflects actual dissimilarity magnitude ---
    # sklearn normalises the embedding to a fixed scale regardless of how
    # dissimilar the samples are.  Multiplying by (mean_dissimilarity / std)
    # restores a meaningful axis scale without altering the topology.
    mean_dissim = float(np.mean(dissimilarity_matrix[upper]))
    spread = float(np.std(embedding))
    if spread > 0:
        embedding = embedding * (mean_dissim / spread)

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
    dissimilarity_matrix, prob_distros, c_distros = _compute_dissimilarity_matrix(samples, metric)
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