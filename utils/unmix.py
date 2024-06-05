import utils.test
import random
import numpy as np
import pandas as pd
from utils import graph


def do_monte_carlo(samples, num_trials=10000):
    sink_sample = samples[0]
    source_samples = samples[1:]
    trials = [None] * num_trials

    sink_sample.replace_bandwidth(10)

    for source_sample in source_samples:
        source_sample.replace_bandwidth(10)

    sink_kde = graph.kde_function(sink_sample)[1]
    source_kdes = [graph.kde_function(source_sample)[1] for source_sample in source_samples]

    for i in range(0, num_trials):
        trial = UnmixingTrial(sink_kde, source_kdes)
        trials[i] = trial

    sorted_trials = sorted(trials, key=lambda x: x.r2_val, reverse=True)
    top_trials = get_percent_of_array(sorted_trials, 1)
    random_configurations = [trial.random_configuration for trial in top_trials]

    source_contributions = np.average(random_configurations, axis=0)*100
    source_std = np.std(random_configurations, axis=0)*100

    contribution_table_d = build_contribution_table(source_samples, source_contributions, source_std, test_type="r2")
    return contribution_table_d


def get_percent_of_array(arr, percentage):
    array_length = len(arr)
    num_elements_in_percentage = int(np.round(array_length * percentage / 100, decimals=0))
    elements_returned = arr[:num_elements_in_percentage]
    return elements_returned


def build_contribution_table(samples, percent_contributions, standard_deviation, test_type="r2"):
    sample_names = [sample.name for sample in samples]
    data = {
        "Sample Name": sample_names,
        f"% Contribution ({test_type} test)": percent_contributions,
        "Standard Deviation": standard_deviation
    }
    df = pd.DataFrame(data)
    df.columns.name = "-"
    output = df.to_html(classes="table table-bordered table-striped", justify="center").replace('<th>','<th style = "background-color: White;">').replace('<td>','<td style = "background-color: White;">')
    return output


class UnmixingTrial:
    def __init__(self, sink_kde, source_kdes):
        self.sink_kde = sink_kde
        self.source_kdes = source_kdes
        rands, d_val, v_val, r2_val = self.__do_trial()
        self.random_configuration = rands
        self.d_val = d_val
        self.v_val = v_val
        self.r2_val = r2_val

    def __do_trial(self):
        sink_kde = self.sink_kde
        source_kdes = self.source_kdes

        num_sources = len(source_kdes)
        rands = self.__make_cumulative_random(num_sources)

        model_kde = np.zeros_like(sink_kde)
        for j, source_kde in enumerate(source_kdes):
            scale_weight = rands[j]
            for k in range(len(sink_kde)):
                model_kde[k] += source_kde[k] * scale_weight

        d_val = None
        v_val = None
        r2_val = utils.test.r2(sink_kde, model_kde)
        return rands, d_val, v_val, r2_val

    @staticmethod
    def __make_cumulative_random(num_samples):
        rands = [random.random() for _ in range(num_samples)]
        total = sum(rands)
        normalized_rands = [rand / total for rand in rands]
        return normalized_rands

    @staticmethod
    def __get_percent_of_array(arr, percentage):
        array_length = len(arr)
        num_elements_in_percentage = int(np.round(array_length * percentage / 100, decimals=0))
        print(num_elements_in_percentage)
        elements_returned = arr[:num_elements_in_percentage]
        return elements_returned
