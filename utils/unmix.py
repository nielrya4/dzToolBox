import utils.test
import random
import numpy as np
import pandas as pd
from utils import graph
import matplotlib.pyplot as plt
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor


def do_monte_carlo(samples, num_trials=10000):
    sink_sample = samples[0]
    source_samples = samples[1:]

    sink_sample.replace_bandwidth(10)
    for source_sample in source_samples:
        source_sample.replace_bandwidth(10)

    sink_kde = graph.kde_function(sink_sample)[1]
    source_kdes = [graph.kde_function(source_sample)[1] for source_sample in source_samples]

    with ProcessPoolExecutor() as executor:
        trials = list(executor.map(create_trial, [(sink_kde, source_kdes)] * num_trials))

    sorted_trials = sorted(trials, key=lambda x: x.r2_val, reverse=True)
    top_trials = sorted_trials[:10]
    top_kdes = [trial.model_kde for trial in top_trials]
    random_configurations = [trial.random_configuration for trial in top_trials]

    source_contributions = np.average(random_configurations, axis=0) * 100
    source_std = np.std(random_configurations, axis=0) * 100

    contribution_table = build_contribution_table(source_samples, source_contributions, source_std, test_type="r2")
    contribution_graph = build_contribution_graph(source_samples, source_contributions, source_std, test_type="r2")
    top_trials_graph = build_top_trials_graph(sink_kde, top_kdes)
    return contribution_table, contribution_graph, top_trials_graph

def create_trial(args):
    sink_kde, source_kdes = args
    return UnmixingTrial(sink_kde, source_kdes)

def build_contribution_table(samples, percent_contributions, standard_deviation, test_type="r2"):
    sample_names = [sample.name for sample in samples]
    data = {
        "Sample Name": sample_names,
        f"% Contribution ({test_type} test)": percent_contributions,
        "Standard Deviation": standard_deviation
    }
    df = pd.DataFrame(data)
    df.columns.name = "-"
    output = df.to_html(classes="table table-bordered table-striped", justify="center").replace('<th>', '<th style="background-color: White;">').replace('<td>', '<td style="background-color: White;">')
    return output

def build_contribution_graph(samples, percent_contributions, standard_deviations, test_type="r2"):
    sample_names = [sample.name for sample in samples]
    x = range(len(samples))
    y = percent_contributions
    e = standard_deviations

    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=e, linestyle="none", marker='.')

    ax.set_title("Relative Contribution Graph")
    ax.set_xticks(x)
    ax.set_xticklabels(sample_names, rotation=45, ha='right')
    plt.tight_layout()
    image_buffer = BytesIO()
    fig.savefig(image_buffer, format="svg", bbox_inches="tight")
    image_buffer.seek(0)
    plotted_graph = image_buffer.getvalue().decode("utf-8")
    plt.close(fig)
    return plotted_graph

def build_top_trials_graph(sink_kde, model_kdes):
    x = np.linspace(0, 4000, 1000).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    for i, model_kde in enumerate(model_kdes):
        ax.plot(x, model_kde, 'c-', label="Top Trials" if i == 0 else "_Top Trials")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.plot(x, sink_kde, 'b-', label="Sink Sample")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title("Top Trials Graph")
    plt.tight_layout()
    image_buffer = BytesIO()
    fig.savefig(image_buffer, format="svg", bbox_inches="tight")
    image_buffer.seek(0)
    plotted_graph = image_buffer.getvalue().decode("utf-8")
    plt.close(fig)
    return plotted_graph

class UnmixingTrial:
    def __init__(self, sink_kde, source_kdes):
        self.sink_kde = sink_kde
        self.source_kdes = source_kdes
        self.random_configuration, self.model_kde, _, _, self.r2_val = self.__do_trial()

    def __do_trial(self):
        sink_kde = self.sink_kde
        source_kdes = self.source_kdes

        num_sources = len(source_kdes)
        rands = self.__make_cumulative_random(num_sources)

        model_kde = np.zeros_like(sink_kde)
        for j, source_kde in enumerate(source_kdes):
            model_kde += source_kde * rands[j]

        r2_val = utils.test.r2(sink_kde, model_kde)
        return rands, model_kde, None, None, r2_val

    @staticmethod
    def __make_cumulative_random(num_samples):
        rands = np.random.random(num_samples)
        return rands / rands.sum()
