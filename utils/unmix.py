import utils.test
import random
import numpy as np
import pandas as pd
from utils import graph
import matplotlib.pyplot as plt
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor
import base64


def do_monte_carlo(samples, num_trials=10000, test_type="r2"):
    sink_sample = samples[0]
    source_samples = samples[1:]

    sink_sample.replace_bandwidth(10)
    for source_sample in source_samples:
        source_sample.replace_bandwidth(10)

    if test_type == "r2":
        sink_line = graph.kde_function(sink_sample)[1]
        source_lines = [graph.kde_function(source_sample)[1] for source_sample in source_samples]
    elif test_type == "ks" or test_type == "kuiper":
        sink_line = graph.cdf_function(sink_sample)[1]
        source_lines = [graph.cdf_function(source_sample)[1] for source_sample in source_samples]
    else:
        sink_line = graph.kde_function(sink_sample)[1]
        source_lines = [graph.kde_function(source_sample)[1] for source_sample in source_samples]

    with ProcessPoolExecutor() as executor:
        trials = list(executor.map(create_trial, [(sink_line, source_lines, test_type)] * num_trials))
    if test_type == "r2":
        sorted_trials = sorted(trials, key=lambda x: x.test_val, reverse=True)
    elif test_type == "ks" or test_type == "kuiper":
        sorted_trials = sorted(trials, key=lambda x: x.test_val, reverse=False)
    else:
        sorted_trials = sorted(trials, key=lambda x: x.test_val, reverse=True)

    top_trials = sorted_trials[:10]
    for trial in top_trials:
        print(trial.test_val)
    top_lines = [trial.model_line for trial in top_trials]
    random_configurations = [trial.random_configuration for trial in top_trials]

    source_contributions = np.average(random_configurations, axis=0) * 100
    source_std = np.std(random_configurations, axis=0) * 100

    contribution_table = build_contribution_table(source_samples, source_contributions, source_std, test_type=test_type)
    contribution_graph = build_contribution_graph(source_samples, source_contributions, source_std, test_type=test_type, download_link=True)
    top_trials_graph = build_top_trials_graph(sink_line, top_lines, download_link=True)
    return contribution_table, contribution_graph, top_trials_graph


def create_trial(args):
    sink_line, source_lines, test_type = args
    return UnmixingTrial(sink_line, source_lines, test_type=test_type)


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


def build_contribution_graph(samples, percent_contributions, standard_deviations, test_type="r2", download_link=False):
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

    encoded_data = base64.b64encode(plotted_graph.encode('utf-8')).decode('utf-8')
    if download_link:
        html = f'<div><img src="data:image/svg+xml;base64,{encoded_data}" download="image.svg"/> <br /> <a href="data:image/svg+xml;base64,{encoded_data}" download="image.svg">Download SVG</a></div>'
    else:
        html = f'<div><img src="data:image/svg+xml;base64,{encoded_data}" download="image.svg"/></div>'
    return html

def build_top_trials_graph(sink_line, model_lines, download_link=False):
    x = np.linspace(0, 4000, 1000).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    for i, model_kde in enumerate(model_lines):
        ax.plot(x, model_kde, 'c-', label="Top Trials" if i == 0 else "_Top Trials")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.plot(x, sink_line, 'b-', label="Sink Sample")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title("Top Trials Graph")
    plt.tight_layout()
    image_buffer = BytesIO()
    fig.savefig(image_buffer, format="svg", bbox_inches="tight")
    image_buffer.seek(0)
    plotted_graph = image_buffer.getvalue().decode("utf-8")
    plt.close(fig)

    encoded_data = base64.b64encode(plotted_graph.encode('utf-8')).decode('utf-8')
    if download_link:
        html = f'<div><img src="data:image/svg+xml;base64,{encoded_data}" download="image.svg"/> <br /> <a href="data:image/svg+xml;base64,{encoded_data}" download="image.svg">Download SVG</a></div>'
    else:
        html = f'<div><img src="data:image/svg+xml;base64,{encoded_data}" download="image.svg"/></div>'
    return html

class UnmixingTrial:
    def __init__(self, sink_line, source_lines, test_type="r2"):
        self.sink_line = sink_line
        self.source_lines = source_lines
        self.test_type = test_type
        self.random_configuration, self.model_line, self.test_val = self.__do_trial()

    def __do_trial(self):
        sink_line = self.sink_line
        source_lines = self.source_lines

        num_sources = len(source_lines)
        rands = self.__make_cumulative_random(num_sources)

        model_line = np.zeros_like(sink_line)
        for j, source_line in enumerate(source_lines):
            model_line += source_line * rands[j]

        if self.test_type == "r2":
            val = utils.test.r2(sink_line, model_line)
        elif self.test_type == "ks":
            val = utils.test.ks(sink_line, model_line)
        elif self.test_type == "kuiper":
            val = utils.test.kuiper(sink_line, model_line)
        else:
            val = utils.test.r2(sink_line, model_line)

        return rands, model_line, val

    @staticmethod
    def __make_cumulative_random(num_samples):
        rands = [random.random() for _ in range(num_samples)]
        total = sum(rands)
        normalized_rands = [rand / total for rand in rands]
        return normalized_rands
