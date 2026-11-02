import numpy as np
import pandas as pd
from dz_lib.univariate.distributions import Distribution, distribution_graph
from dz_lib.utils import fonts
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


class Contribution:
    def __init__(self, name: str, contribution: float, standard_deviation: float):
        self.name = name
        self.contribution = contribution
        self.standard_deviation = standard_deviation

def _misfit_function(metric: str):
    """Vectorised misfit (lower is better) of model curves (rows) against the sink curve."""
    def cross_correlation(models, sink):
        sink_c = sink - sink.mean()
        models_c = models - models.mean(axis=1, keepdims=True)
        r = (models_c @ sink_c) / np.sqrt(np.sum(models_c ** 2, axis=1) * np.sum(sink_c ** 2))
        return 1.0 - r ** 2

    def ks(models, sink):
        return np.max(np.abs(models - sink), axis=1)

    def kuiper(models, sink):
        diff = models - sink
        return np.maximum(diff.max(axis=1), 0.0) + np.maximum((-diff).max(axis=1), 0.0)

    try:
        return {"cross_correlation": cross_correlation, "ks": ks, "kuiper": kuiper}[metric]
    except KeyError:
        raise ValueError(f"Unknown metric '{metric}'")


def _refine(starts, sources, sink, misfit):
    """Constrained optimisation of the mixing proportions (SLSQP, as in detritalPy).

    KS D and Kuiper V are maxima, so their misfit surface has many shallow local minima;
    starting from several good trials (and from equal proportions) and keeping the lowest
    misfit makes the answer stable from run to run.
    """
    n = starts.shape[1]
    objective = lambda w: float(misfit(w[None, :] @ sources, sink)[0])
    best_w, best_val = starts[0], objective(starts[0])
    for start in list(starts) + [np.full(n, 1.0 / n)]:
        result = minimize(objective, start, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                          constraints={"type": "eq", "fun": lambda w: 1.0 - np.sum(w)},
                          options={"maxiter": 10000})
        w = np.clip(result.x, 0.0, None)
        w = w / w.sum()
        val = objective(w)
        if val < best_val:
            best_w, best_val = w, val
    return best_w


def _dzmix_weights(n_trials: int, n_sources: int, rng) -> np.ndarray:
    """Random proportions as DZmix generates them (Sundell & Saylor, 2017; the Monte Carlo
    loop of Sundell_Saylor_2017_Inverse_unmixing_model_v_1_1_02.m).

    Breakpoints on [0, 1] are placed in random order, each at least its share of an even
    split along the line, and the gaps between them, shuffled, are the proportions. This
    favours lopsided mixtures: with 10 sources about 74% of trials give some source more
    than 30%, against about 39% for a uniform (Dirichlet) draw.
    """
    weights = np.empty((n_trials, n_sources))
    for trial in range(n_trials):
        order = rng.permutation(np.arange(1, n_sources))
        breakpoints = np.zeros(n_sources + 1)
        breakpoints[n_sources] = 1.0
        floors = order * (1.0 / n_sources)   # as DZmix computes it (samples .* 1/N)
        for i, k in enumerate(order):
            lower = breakpoints[:k].max()
            upper_candidates = breakpoints[k:]
            upper = upper_candidates[upper_candidates != 0].min()
            start = floors[i] if lower < floors[i] else lower
            breakpoints[k] = start + rng.random() * (upper - start)
        weights[trial] = np.diff(breakpoints)[rng.permutation(n_sources)]
    return weights


# How random mixing proportions are drawn for the Monte Carlo trials
SAMPLERS = {
    "dirichlet": lambda n_trials, n_sources, rng: rng.dirichlet(np.ones(n_sources), size=n_trials),
    "dzmix": _dzmix_weights,
}


def monte_carlo_model(sink_distribution: Distribution, source_distributions: [Distribution], n_trials: int = 10000,
                      metric: str = "cross_correlation", n_top: int = 10, refine: bool = True, seed=None,
                      sampler: str = "dirichlet"):
    """Unmix a sink distribution into source contributions (after Sundell & Saylor, 2017).

    Random mixing proportions are drawn uniformly over all possible mixtures
    (sampler='dirichlet') or with DZmix's stick-breaking generator (sampler='dzmix', see
    _dzmix_weights); each is scored against the sink with the chosen metric (1 - R^2 on PDPs/KDEs, or KS D /
    Kuiper V on CDFs).

    refine=True
        The best trials are refined by constrained optimisation (SLSQP, bounds 0-1, sum 1),
        like DZmix's default "minimum function search" (fmincon from the best Monte Carlo
        fits) and detritalPy. The contributions are the single best fit.
    refine=False
        Monte Carlo only, as in Sundell & Saylor (2017): the contributions are the mean of
        the `n_top` best trials.

    Returns
    -------
    source_contributions : percent contribution of each source
    source_std : standard deviation of the `n_top` best Monte Carlo trials, in percent
    top_distributions : model curves of the `n_top` best trials (refined best fit first)
    """
    sink = np.asarray(sink_distribution.y_values, dtype=float)
    sources = np.array([np.asarray(d.y_values, dtype=float) for d in source_distributions])
    misfit = _misfit_function(metric)
    rng = np.random.default_rng(seed)

    try:
        draw = SAMPLERS[sampler]
    except KeyError:
        raise ValueError(f"Unknown sampler '{sampler}'")
    weights = draw(int(n_trials), len(sources), rng)
    scores = np.empty(len(weights))
    chunk = 1000  # bounds memory: chunk x curve length model curves at a time
    for start in range(0, len(weights), chunk):
        scores[start:start + chunk] = misfit(weights[start:start + chunk] @ sources, sink)
    top_weights = weights[np.argsort(scores, kind="stable")[:n_top]]

    if refine:
        best = _refine(top_weights, sources, sink, misfit)
        model_weights = [best] + list(top_weights[1:])
    else:
        best = top_weights.mean(axis=0)
        model_weights = list(top_weights)
    x_values = sink_distribution.x_values
    top_distributions = [Distribution(f"Top_Trial_{i + 1}", x_values, w @ sources)
                         for i, w in enumerate(model_weights)]
    return best * 100, top_weights.std(axis=0) * 100, top_distributions


def relative_contribution_graph(
        contributions: [Contribution],
        title: str = "Relative Contribution Graph",
        font_path: str = None,
        font_size: float = 12,
        fig_width: float = 9,
        fig_height: float = 7,
):
    sample_names = [contribution.name for contribution in contributions]
    x = range(len(contributions))
    y = [contribution.contribution for contribution in contributions]
    e = [contribution.standard_deviation for contribution in contributions]
    if font_path:
        font = fonts.get_font(font_path)
    else:
        font = fonts.get_default_font()
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100, squeeze=True)
    ax.errorbar(x, y, yerr=e, linestyle="none", marker='.')
    ax.set_title(title, fontsize=font_size * 2, fontproperties=font)
    ax.set_xticks(x)
    ax.set_xticklabels(sample_names, rotation=45, ha='right', fontsize=font_size, fontproperties=font)
    plt.tight_layout()
    plt.close()
    return fig


def relative_contribution_table(
        contributions: [Contribution],
        metric: str = "cross_correlation",
        title=f"Relative Contribution Table",
        refined: bool = True,
        n_top: int = 10
    ):
    sample_names = [contribution.name for contribution in contributions]
    percent_contributions = [contribution.contribution for contribution in contributions]
    standard_deviations = [contribution.standard_deviation for contribution in contributions]
    estimate = "optimised best fit" if refined else f"mean of best {n_top} trials"
    data = {
        f"% Contribution ({estimate}, metric={metric})": percent_contributions,
        f"Std. Dev. of best {n_top} trials": standard_deviations
    }
    indices = [f"{name}" for name in sample_names]
    df = pd.DataFrame(data, index=indices)
    df.style.set_table_attributes("style='display:inline'").set_caption(title)
    df = df.rename_axis(columns="Sample Name")
    return df


def top_trials_graph(
        sink_distribution: Distribution,
        model_distributions: [Distribution],
        x_min: float = 0,
        x_max: float = 4500,
        title: str = "Top Trials Graph",
        font_path: str = None,
        font_size: float = 12,
        fig_width: float = 9,
        fig_height: float = 7,
    ):
    # Plot trial distributions first (so sink appears on top)
    trial_distributions = model_distributions
    
    # Create custom colormap using original colors: cyan for trials, blue for sink
    num_distributions = len(trial_distributions) + 1
    colors = ['cyan'] * len(trial_distributions)  # Cyan for trial lines
    colors.append('blue')  # Blue for sink sample
    custom_colormap = ListedColormap(colors)
    
    # Combine trial distributions first, then sink (for plotting order)
    all_distributions = trial_distributions + [sink_distribution]

    # Use the distributions module's graph function without legend
    fig = distribution_graph(
        distributions=all_distributions,
        x_min=x_min,
        x_max=x_max,
        title=title,
        font_path=font_path,
        font_size=font_size,
        fig_width=fig_width,
        fig_height=fig_height,
        legend=False,
        stacked=False,
        color_map=custom_colormap
    )
    
    # Add custom legend
    ax = fig.get_axes()[0]
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='cyan', label='Top Trials'),
        Line2D([0], [0], color='blue', label=sink_distribution.name)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=font_size)
    
    return fig