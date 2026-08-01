import numpy as np
import pandas as pd
from dz_lib.univariate import metrics
from dz_lib.univariate.distributions import Distribution, distribution_graph
from dz_lib.utils import fonts
import random
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


class Contribution:
    def __init__(self, name: str, contribution: float, standard_deviation: float):
        self.name = name
        self.contribution = contribution
        self.standard_deviation = standard_deviation

def monte_carlo_model(sink_distribution: Distribution, source_distributions: [Distribution], n_trials: int=10000, metric: str="cross_correlation"):
    sink_y_values = sink_distribution.y_values
    sources_y_values = [dist.y_values for dist in source_distributions]
    trials = [create_trial((sink_y_values, sources_y_values, metric)) for _ in range(n_trials)]
    if metric == "cross_correlation":
        sorted_trials = sorted(trials, key=lambda x: x.test_val, reverse=True)
    elif metric == "ks" or metric == "kuiper":
        sorted_trials = sorted(trials, key=lambda x: x.test_val, reverse=False)
    else:
        raise ValueError(f"Unknown metric '{metric}'")
    top_trials = sorted_trials[:10]
    top_lines = [trial.model_line for trial in top_trials]
    
    # Convert top lines back to Distribution objects
    x_values = sink_distribution.x_values
    top_distributions = [Distribution(f"Top_Trial_{i+1}", x_values, y_values) 
                        for i, y_values in enumerate(top_lines)]
    
    random_configurations = [trial.random_configuration for trial in top_trials]
    source_contributions = np.average(random_configurations, axis=0) * 100
    source_std = np.std(random_configurations, axis=0) * 100
    return source_contributions, source_std, top_distributions

def create_trial(args):
    sink_y_values, sources_y_values, test_type = args
    return UnmixingTrial(sink_y_values, sources_y_values, metric=test_type)

class UnmixingTrial:
    def __init__(self, sink_line: [float], source_lines: [[float]], metric: str="cross_correlation"):
        self.sink_line = sink_line
        self.source_lines = source_lines
        self.metric = metric
        self.random_configuration, self.model_line, self.test_val = self.__do_trial()

    def __do_trial(self):
        sink_line = self.sink_line
        source_lines = self.source_lines
        n_sources = len(source_lines)
        rands = self.__make_cumulative_random(n_sources)
        model_line = np.zeros_like(sink_line)
        for j, source_line in enumerate(source_lines):
            model_line += source_line * rands[j]
        if self.metric == "cross_correlation":
            val = metrics.r2(sink_line, model_line)
        elif self.metric == "ks":
            val = metrics.ks(sink_line, model_line)
        elif self.metric == "kuiper":
            val = metrics.kuiper(sink_line, model_line)
        else:
            raise ValueError(f"Unknown metric '{self.metric}'")
        return rands, model_line, val

    @staticmethod
    def __make_cumulative_random(num_samples):
        rands = [random.random() for _ in range(num_samples)]
        total = sum(rands)
        normalized_rands = [rand / total for rand in rands]
        return normalized_rands


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
        title=f"Relative Contribution Table"
    ):
    sample_names = [contribution.name for contribution in contributions]
    percent_contributions = [contribution.contribution for contribution in contributions]
    standard_deviations = [contribution.standard_deviation for contribution in contributions]
    data = {
        f"% Contribution (metric={metric})": percent_contributions,
        "Standard Deviation": standard_deviations
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