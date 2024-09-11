import numpy as np
from sklearn.neighbors import KernelDensity
from utils import test
from sklearn.manifold import MDS as MultidimensionalScaling
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import secrets


class Graph:
    def __init__(self, samples, title, graph_type, stacked=False, legend=True, kde_bandwidth=10):
        self.samples = samples
        self.title = title
        self.graph_type = graph_type
        self.legend = legend
        self.stacked = stacked
        self.kde_bandwidth = kde_bandwidth

    def generate_fig(self):
        gtype = self.graph_type
        stacked = self.stacked
        samples = self.samples
        title = self.title
        kde_bandwidth = self.kde_bandwidth
        if gtype == 'kde':
            return kde_graph(samples, title, stacked=stacked, kde_bandwidth=kde_bandwidth)
        elif gtype == 'pdp':
            return pdp_graph(samples, title, stacked=stacked)
        elif gtype == 'cdf':
            return cdf_graph(samples, title)
        elif gtype == 'sim_mds':
            return mds_graph(samples, title, 'similarity')
        elif gtype == 'ks_mds':
            return mds_graph(samples, title, 'ks')
        elif gtype == 'kuiper_mds':
            return mds_graph(samples, title, 'kuiper')
        elif gtype == 'r2_mds':
            return mds_graph(samples, title, 'r2')

    def generate_svg(self):
        fig = self.generate_fig()
        image_buffer = BytesIO()
        fig.savefig(image_buffer, format="svg", bbox_inches="tight")
        image_buffer.seek(0)
        plotted_graph = image_buffer.getvalue().decode("utf-8")
        plt.close(fig)
        return plotted_graph

    def generate_html(self, output_id, actions_button=False):
        svg = self.generate_svg()
        encoded_data = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
        if actions_button:
            html = f"""
                    <div>
                        <img src="data:image/svg+xml;base64,{encoded_data}" download="image.svg"/>
                        <div class="dropdown show">
                            <a class="btn btn-secondary dropdown-toggle" href="#" role="button" id="{output_id}_dropdown" data-bs-toggle="dropdown" aria-expanded="false">
                                Actions
                            </a>
                            <div class="dropdown-menu" aria-labelledby="{output_id}_dropdown">
                                <a class="dropdown-item" href="data:image/svg+xml;base64,{encoded_data}" download="image.svg">Download As SVG</a>
                                <a class="dropdown-item" href="#" data-hx-post="/delete_output/{output_id}" data-hx-target="#outputs_container" data-hx-swap="innerHTML">Delete This Output</a>
                            </div>
                        </div>
                    </div>"""
        else:
            html = f'<div><img src="data:image/svg+xml;base64,{encoded_data}" download="image.svg"/></div>'
        return html

    def generate_pdf(self):
        fig = self.generate_fig()
        image_buffer = BytesIO()
        fig.savefig(image_buffer, format="pdf", bbox_inches="tight")
        image_buffer.seek(0)
        plotted_graph = image_buffer.getvalue().decode("utf-8")
        plt.close(fig)
        return plotted_graph

    def generate_eps(self):
        fig = self.generate_fig()
        image_buffer = BytesIO()
        fig.savefig(image_buffer, format="eps", bbox_inches="tight")
        image_buffer.seek(0)
        plotted_graph = image_buffer.getvalue().decode("utf-8")
        plt.close(fig)
        return plotted_graph

    def generate_png(self):
        fig = self.generate_fig()
        image_buffer = BytesIO()
        fig.savefig(image_buffer, format="png", bbox_inches="tight")
        image_buffer.seek(0)
        plotted_graph = image_buffer.getvalue().decode("utf-8")
        plt.close(fig)
        return plotted_graph


# Graph functions that output ([x], [y]):
def kde_function(sample, num_steps=1000, x_min=0, x_max=4000, kde_bandwidth=10):
    grains = sample.grains
    sample.replace_bandwidth(kde_bandwidth)
    bandwidths = np.abs([grain.uncertainty for grain in grains])
    mean_bandwidth = np.mean(bandwidths)
    ages = np.array([grain.age for grain in grains])
    x = np.linspace(x_min, x_max, num_steps).reshape(-1, 1)
    kde = KernelDensity(bandwidth=mean_bandwidth, kernel='gaussian')
    kde.fit(ages.reshape(-1, 1))
    log_dens = kde.score_samples(x)
    y = np.exp(log_dens)
    y_normalized = y / np.sum(y)
    return x.flatten(), y_normalized


def cdf_function(sample):
    _, y_values = kde_function(sample)
    cdf_values = np.cumsum(y_values)
    return range(0, 1001), cdf_values


def pdp_function(sample, num_steps=1000, x_min=0, x_max=4000):
    x = np.linspace(x_min, x_max, num_steps)
    y = np.zeros_like(x)
    ages = [grain.age for grain in sample.grains]
    bandwidths = [grain.uncertainty for grain in sample.grains]
    for i in range(len(ages)):
        kernel_sum = np.zeros(num_steps)
        s = bandwidths[i]
        kernel_sum += (1.0 / (np.sqrt(2 * np.pi) * s)) * np.exp(-(x - float(ages[i])) ** 2 / (2 * float(s) ** 2))
        y += kernel_sum
    y /= np.sum(y)
    return x, y


# Graph functions that output a fig object
def kde_graph(samples, title, stacked=False, kde_bandwidth=10):
    x_max = get_x_max(samples)
    x_min = get_x_min(samples)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    if not stacked:
        for i, sample in enumerate(samples):
            header = sample.name
            x, y = kde_function(sample, x_max=x_max, x_min=x_min, kde_bandwidth=kde_bandwidth)
            ax.plot(x, y, label=header)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    else:
        if len(samples) == 1:
            fig, ax = plt.subplots(nrows=1, figsize=(9, 6), dpi=100, squeeze=False)
            for i, sample in enumerate(samples):
                header = sample.name
                x, y = kde_function(sample, x_max=x_max, x_min=x_min, kde_bandwidth=kde_bandwidth)
                ax[0, 0].plot(x, y, label=header)
                ax[0, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        else:
            fig, ax = plt.subplots(nrows=len(samples), figsize=(9, 6), dpi=100, squeeze=False)
            for i, sample in enumerate(samples):
                header = sample.name
                x, y = kde_function(sample, x_max=x_max, x_min=x_min, kde_bandwidth=kde_bandwidth)
                ax[i, 0].plot(x, y, label=header)
                ax[i, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.suptitle(title if not None else "Kernel Density Estimate")
    fig.text(0.5, 0.01, 'Age (Ma)', ha='center', va='center', fontsize=12)
    fig.text(0.01, 0.5, 'Probability Differential', va='center', rotation='vertical', fontsize=12)
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 1])
    return fig


def pdp_graph(samples, title, stacked=False):
    x_max = get_x_max(samples)
    x_min = get_x_min(samples)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    if not stacked:
        for i, sample in enumerate(samples):
            header = sample.name
            x, y = pdp_function(sample, x_max=x_max, x_min=x_min)
            ax.plot(x, y, label=header)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    else:
        if len(samples) == 1:
            fig, ax = plt.subplots(nrows=1, figsize=(8, 6), dpi=100, squeeze=False)
            for i, sample in enumerate(samples):
                header = sample.name
                x, y = pdp_function(sample, x_max=x_max, x_min=x_min)
                ax[0, 0].plot(x, y, label=header)
                ax[0, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        else:
            fig, ax = plt.subplots(nrows=len(samples), figsize=(8, 6), dpi=100, squeeze=False)
            for i, sample in enumerate(samples):
                header = sample.name
                x, y = pdp_function(sample, x_max=x_max, x_min=x_min)
                ax[i, 0].plot(x, y, label=header)
                ax[i, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.suptitle(title if not None else "Kernel Density Estimate")
    fig.tight_layout()
    return fig


def cdf_graph(samples, title):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    for sample in samples:
        header = sample.name
        bins_count, cdf_values = cdf_function(sample)
        ax.plot(bins_count[1:], cdf_values, label=header)
    ax.set_title(title if title is not None else "Cumulative Distribution Function")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    return fig


def mds_graph(samples, title, mds_type, kde_bandwidth=10):
    num_samples = len(samples)
    dissimilarity_matrix = np.zeros((num_samples, num_samples))
    sample_names = [sample.name for sample in samples]
    sample_kdes = [kde_function(sample, kde_bandwidth=kde_bandwidth)[1] for sample in samples]
    sample_cdfs = [cdf_function(sample)[1] for sample in samples]

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if mds_type == 'similarity':
                dissimilarity_matrix[i, j] = test.dis_similarity(sample_kdes[i], sample_kdes[j])
            elif mds_type == 'ks':
                dissimilarity_matrix[i, j] = test.dis_ks(sample_cdfs[i], sample_cdfs[j])
            elif mds_type == 'kuiper':
                dissimilarity_matrix[i, j] = test.dis_kuiper(sample_cdfs[i], sample_cdfs[j])
            elif mds_type == 'r2':
                dissimilarity_matrix[i, j] = test.dis_r2(sample_kdes[i], sample_kdes[j])
            dissimilarity_matrix[j, i] = dissimilarity_matrix[i, j]

    embedding = MultidimensionalScaling(n_components=2, dissimilarity='precomputed')
    scaled_mds_result = embedding.fit_transform(dissimilarity_matrix)

    viridis = plt.cm.get_cmap('gist_ncar', num_samples)
    colors = viridis(np.linspace(0, 1, num_samples))
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    for i, (x, y) in enumerate(scaled_mds_result):
        ax.scatter(x, y, color=colors[i])
        ax.text(x, y + 0.005, sample_names[i], fontsize=8, ha='center', va='center')

    for i, (x, y) in enumerate(scaled_mds_result):
        ax.text(x, y + 0.005, sample_names[i], fontsize=8, ha='center', va='center')

    for i in range(num_samples):
        distance = float('inf')  # Initialize distance to positive infinity
        nearest_sample = None

        for j in range(num_samples):
            if i != j:  # Exclude the sample itself
                if mds_type == 'similarity':
                    dissimilarity = test.dis_similarity(sample_kdes[i], sample_kdes[j])
                elif mds_type == 'ks':
                    dissimilarity = test.dis_ks(sample_cdfs[i], sample_cdfs[j])
                elif mds_type == 'kuiper':
                    dissimilarity = test.dis_kuiper(sample_cdfs[i], sample_cdfs[j])
                elif mds_type == 'r2':
                    dissimilarity = test.dis_r2(sample_kdes[i], sample_kdes[j])
                else:
                    dissimilarity = test.dis_similarity(sample_kdes[i], sample_kdes[j])
                if dissimilarity < distance:
                    distance = dissimilarity
                    nearest_sample = samples[j]

        if nearest_sample is not None:
            x1, y1 = scaled_mds_result[i]
            x2, y2 = scaled_mds_result[samples.index(nearest_sample)]
            ax.plot([x1, x2], [y1, y2], 'k--', linewidth=0.5)

    stress = embedding.stress_

    fig.suptitle(title if title is not None else f"Multidimensional Scaling Plot (Stress: {np.round(stress, decimals=6)})")
    fig.text(0.5, 0.01, 'Dimension 1', ha='center', va='center', fontsize=12)
    fig.text(0.01, 0.5, 'Dimension 2', va='center', rotation='vertical', fontsize=12)
    fig.tight_layout()
    return fig


# Extra Graph Utils:
def get_x_max(samples):
    x_max = 0
    for sample in samples:
        for grain in sample.grains:
            if grain.age + grain.uncertainty > x_max:
                x_max = grain.age + grain.uncertainty
    return x_max


def get_x_min(samples):
    x_min = 0
    for sample in samples:
        for grain in sample.grains:
            if grain.age - grain.uncertainty < x_min:
                x_min = grain.age - grain.uncertainty
    return x_min
