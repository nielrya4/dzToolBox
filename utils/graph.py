import numpy as np
from utils import test, fonts, graph_3d
from sklearn.manifold import MDS as MultidimensionalScaling
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import cairosvg
from scipy.interpolate import interp1d

class Graph:
    def __init__(self,
                 samples,
                 title,
                 graph_type,
                 stacked=False,
                 legend=True,
                 min_age=0,
                 max_age=4500,
                 kde_bandwidth=10,
                 color_map='plasma',
                 font_size=12,
                 font_name="ubuntu",
                 fig_width=9,
                 fig_height=7):
        self.samples = samples
        self.title = title
        self.graph_type = graph_type
        self.legend = legend
        self.stacked = stacked
        self.kde_bandwidth = kde_bandwidth
        self.color_map = color_map
        self.font_size = font_size
        self.font_name = font_name
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.min_age = min_age
        self.max_age = max_age

    def generate_fig(self):
        gtype = self.graph_type
        stacked = self.stacked
        samples = self.samples
        title = self.title
        kde_bandwidth = self.kde_bandwidth
        if gtype == 'kde':
            return kde_graph(samples,
                             title,
                             stacked=stacked,
                             kde_bandwidth=kde_bandwidth,
                             legend=self.legend,
                             color_map=self.color_map,
                             font_size=self.font_size,
                             font_name=self.font_name,
                             fig_width=self.fig_width,
                             fig_height=self.fig_height,
                             min_age=self.min_age,
                             max_age=self.max_age)
        elif gtype == 'pdp':
            return pdp_graph(samples,
                             title,
                             stacked=stacked,
                             legend=self.legend,
                             color_map=self.color_map,
                             font_size=self.font_size,
                             font_name=self.font_name,
                             fig_width=self.fig_width,
                             fig_height=self.fig_height,
                             min_age=self.min_age,
                             max_age=self.max_age)
        elif gtype == 'cdf':
            return cdf_graph(samples,
                             title,
                             stacked=stacked,
                             legend=self.legend,
                             color_map=self.color_map,
                             font_size=self.font_size,
                             font_name=self.font_name,
                             fig_width=self.fig_width,
                             fig_height=self.fig_height,
                             min_age=self.min_age,
                             max_age=self.max_age)
        elif gtype == 'sim_mds':
            return mds_graph(samples, title, 'similarity', color_map=self.color_map)
        elif gtype == 'like_mds':
            return mds_graph(samples, title, 'likeness', color_map=self.color_map)
        elif gtype == 'ks_mds':
            return mds_graph(samples, title, 'ks', color_map=self.color_map)
        elif gtype == 'kuiper_mds':
            return mds_graph(samples, title, 'kuiper', color_map=self.color_map)
        elif gtype == 'r2_mds':
            return mds_graph(samples, title, 'r2', color_map=self.color_map)
        elif gtype == "kde2d":
            return graph_3d.kde_graph_2d(samples[-1])
        elif gtype == "heatmap":
            x, y, z, kernel = graph_3d.kde_function_2d(samples[-1])
            return graph_3d.heatmap(x, y, z, title=title, color_map=self.color_map, fig_width=self.fig_width, fig_height=self.fig_height)

    def generate_svg(self):
        fig = self.generate_fig()
        image_buffer = BytesIO()
        fig.savefig(image_buffer, format="svg", bbox_inches="tight")
        image_buffer.seek(0)
        plotted_graph = image_buffer.getvalue().decode("utf-8")
        plt.close(fig)
        return plotted_graph

    def generate_html(self, output_id, actions_button=True):

        if self.graph_type == "heatmap":
            png = self.generate_png()
            encoded_data = base64.b64encode(png).decode("utf-8")
            img_tag = f'<img src="data:image/png;base64,{encoded_data}" download="image.png"/>'
            download_tag = f'<a class="dropdown-item" href="data:image/png;base64,{encoded_data}" download="image.png">Download As PNG</a>'
        else:
            svg = self.generate_svg()
            encoded_data = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
            img_tag = f'<img src="data:image/svg+xml;base64,{encoded_data}" download="image.svg"/>'
            download_tag = f'<a class="dropdown-item" href="data:image/svg+xml;base64,{encoded_data}" download="image.svg">Download As SVG</a>'

        if actions_button:
            html = f"""
                    <div>
                        {img_tag}
                        <div class="dropdown show">
                            <a class="btn btn-secondary dropdown-toggle" href="#" role="button" id="{output_id}_dropdown" data-bs-toggle="dropdown" aria-expanded="false">
                                Actions
                            </a>
                            <div class="dropdown-menu" aria-labelledby="{output_id}_dropdown">
                                {download_tag}
                                <a class="dropdown-item" href="#" data-hx-post="/delete_output/{output_id}" data-hx-target="#outputs_container" data-hx-swap="innerHTML" onclick="show_delete_output_spinner();">Delete This Output</a>
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
        plotted_graph = image_buffer.getvalue()
        plt.close(fig)
        return plotted_graph


# Graph functions that output ([x], [y]):
def kde_function(sample, num_steps=1000, x_min=0, x_max=4500, kde_bandwidth=10):
    x = np.linspace(x_min, x_max, num_steps)
    y = np.zeros_like(x)
    ages = [grain.age for grain in sample.grains]
    for i in range(len(ages)):
        kernel_sum = np.zeros(num_steps)
        s = kde_bandwidth
        kernel_sum += (1.0 / (np.sqrt(2 * np.pi) * s)) * np.exp(-(x - float(ages[i])) ** 2 / (2 * float(s) ** 2))
        kernel_sum /= np.sum(kernel_sum)
        y += kernel_sum
    y /= np.sum(y)
    return x, y


def cdf_function(sample, min_age=0, max_age=4500, kde_bandwidth=10):
    youngest_age = sample.get_youngest_grain().age
    oldest_age = sample.get_oldest_grain().age
    x_kde_values, y_values = kde_function(sample, x_min=youngest_age, x_max=oldest_age)
    cdf_values = np.cumsum(y_values)
    cdf_values = cdf_values / cdf_values[-1]
    x_values = np.linspace(min_age, max_age, 1000)
    x_combined = np.concatenate((
        np.linspace(min_age, youngest_age, num=100),  # 0 to youngest age
        x_kde_values,  # kde values between youngest and oldest ages
        np.linspace(oldest_age, max_age, num=100)  # oldest age to 4500
    ))
    cdf_combined = np.concatenate((
        np.zeros(100),  # CDF is 0 before youngest age
        cdf_values,  # CDF from kde_function
        np.ones(100)  # CDF is 1 after oldest age
    ))
    interpolator = interp1d(x_combined, cdf_combined, kind='linear', bounds_error=False, fill_value=(0, 1))
    cdf_resampled = interpolator(x_values)
    return x_values, cdf_resampled


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
def kde_graph(samples,
              title,
              stacked=False,
              kde_bandwidth=10,
              legend=True,
              min_age=0,
              max_age=4500,
              color_map='plasma',
              font_size=12,
              font_name="ubuntu",
              fig_width=9,
              fig_height=7):
    font = fonts.select_font(font_name)
    title_size = font_size * 2
    x_max = get_x_max(samples)
    x_min = get_x_min(samples)
    num_samples = len(samples)
    colors_map = plt.cm.get_cmap(color_map, num_samples)
    colors = colors_map(np.linspace(0, 1, num_samples))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    if not stacked:
        for i, sample in enumerate(samples):
            header = sample.name
            x, y = kde_function(sample, x_max=x_max, x_min=x_min, kde_bandwidth=kde_bandwidth)
            ax.plot(x, y, label=header, color=colors[i])
            if legend:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    else:
        if len(samples) == 1:
            fig, ax = plt.subplots(nrows=1, figsize=(fig_width, fig_height), dpi=100, squeeze=False)
            for i, sample in enumerate(samples):
                header = sample.name
                x, y = kde_function(sample, x_max=x_max, x_min=x_min, kde_bandwidth=kde_bandwidth)
                ax[0, 0].plot(x, y, label=header)
                if legend:
                    ax[0, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        else:
            fig, ax = plt.subplots(nrows=len(samples), figsize=(fig_width, fig_height), dpi=100, squeeze=False)
            for i, sample in enumerate(samples):
                header = sample.name
                x, y = kde_function(sample, x_max=x_max, x_min=x_min, kde_bandwidth=kde_bandwidth)
                ax[i, 0].plot(x, y, label=header)
                if legend:
                    ax[i, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.suptitle(title if title else "Kernel Density Estimate", fontsize=title_size, fontproperties=font)
    fig.text(0.5, 0.01, 'Age (Ma)', ha='center', va='center', fontsize=font_size, fontproperties=font)
    fig.text(0.01, 0.5, 'Probability Differential', va='center', rotation='vertical', fontsize=font_size, fontproperties=font)
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 1])
    return fig


def pdp_graph(samples,
              title,
              stacked=False,
              legend=True,
              min_age=0,
              max_age=4500,
              color_map='plasma',
              font_size=12,
              font_name="ubuntu",
              fig_width=9,
              fig_height=7):
    font = fonts.select_font(font_name)
    title_size = font_size * 2
    x_max = get_x_max(samples)
    x_min = get_x_min(samples)
    num_samples = len(samples)
    colors_map = plt.cm.get_cmap(color_map, num_samples)
    colors = colors_map(np.linspace(0, 1, num_samples))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    if not stacked:
        for i, sample in enumerate(samples):
            header = sample.name
            x, y = pdp_function(sample, x_max=x_max, x_min=x_min)
            ax.plot(x, y, label=header, color=colors[i])
            if legend:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    else:
        if len(samples) == 1:
            fig, ax = plt.subplots(nrows=1, figsize=(fig_width, fig_height), dpi=100, squeeze=False)
            for i, sample in enumerate(samples):
                header = sample.name
                x, y = pdp_function(sample, x_max=x_max, x_min=x_min)
                ax[0, 0].plot(x, y, label=header)
                if legend:
                    ax[0, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        else:
            fig, ax = plt.subplots(nrows=len(samples), figsize=(fig_width, fig_height), dpi=100, squeeze=False)
            for i, sample in enumerate(samples):
                header = sample.name
                x, y = pdp_function(sample, x_max=x_max, x_min=x_min)
                ax[i, 0].plot(x, y, label=header)
                if legend:
                    ax[i, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.suptitle(title if title else "Probability Density Plit", fontsize=title_size, fontproperties=font)
    fig.text(0.5, 0.01, 'Age (Ma)', ha='center', va='center', fontsize=font_size, fontproperties=font)
    fig.text(0.01, 0.5, 'Probability Differential', va='center', rotation='vertical', fontsize=font_size, fontproperties=font)
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 1])
    return fig


def cdf_graph(samples,
              title=None,
              stacked=False,
              legend=True,
              min_age=0,
              max_age=4500,
              color_map='plasma',
              font_size=12,
              font_name="ubuntu",
              fig_width=9,
              fig_height=7):
    font = fonts.select_font(font_name)
    title_size = font_size * 2
    x_max = get_x_max(samples)
    x_min = get_x_min(samples)
    num_samples = len(samples)
    colors_map = plt.cm.get_cmap(color_map, num_samples)
    colors = colors_map(np.linspace(0, 1, num_samples))

    if not stacked:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        samples.reverse()
        for i, sample in enumerate(samples):
            header = sample.name  # Get sample name for labeling
            x_values, cdf_values = cdf_function(sample, min_age=x_min, max_age=x_max)  # Get x-values and CDF
            ax.plot(x_values, cdf_values, label=header, color=colors[i])
        if legend:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    else:
        fig, ax = plt.subplots(nrows=len(samples), figsize=(fig_width, fig_height), dpi=100, squeeze=False)
        samples.reverse()
        for i, sample in enumerate(samples):
            header = sample.name  # Get sample name for labeling
            x_values, cdf_values = cdf_function(sample, min_age=x_min, max_age=x_max)  # Get x-values and CDF
            ax[i, 0].plot(x_values, cdf_values, label=header)
            if legend:
                ax[i, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.suptitle(title if title else "Cumulative Distribution Function", fontsize=title_size, fontproperties=font)
    fig.text(0.5, 0.01, 'Age (Ma)', ha='center', va='center', fontsize=font_size, fontproperties=font)
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 1])
    return fig

def mds_graph(samples, title, mds_type, kde_bandwidth=10, color_map='plasma'):
    num_samples = len(samples)
    dissimilarity_matrix = np.zeros((num_samples, num_samples))
    sample_names = [sample.name for sample in samples]
    sample_kdes = [kde_function(sample, kde_bandwidth=kde_bandwidth)[1] for sample in samples]
    sample_cdfs = [cdf_function(sample)[1] for sample in samples]

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if mds_type == 'similarity':
                dissimilarity_matrix[i, j] = test.dis_similarity(sample_kdes[i], sample_kdes[j])
            elif mds_type == 'likeness':
                dissimilarity_matrix[i, j] = test.dis_likeness(sample_cdfs[i], sample_cdfs[j])
            elif mds_type == 'ks':
                dissimilarity_matrix[i, j] = test.ks(sample_cdfs[i], sample_cdfs[j])
            elif mds_type == 'kuiper':
                dissimilarity_matrix[i, j] = test.kuiper(sample_cdfs[i], sample_cdfs[j])
            elif mds_type == 'r2':
                dissimilarity_matrix[i, j] = test.dis_r2(sample_kdes[i], sample_kdes[j])
            dissimilarity_matrix[j, i] = dissimilarity_matrix[i, j]

    embedding = MultidimensionalScaling(n_components=2, dissimilarity='precomputed')
    scaled_mds_result = embedding.fit_transform(dissimilarity_matrix)

    colors_map = plt.cm.get_cmap(color_map, num_samples)
    colors = colors_map(np.linspace(0, 1, num_samples))
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
                elif mds_type == 'likeness':
                    dissimilarity = test.dis_likeness(sample_cdfs[i], sample_cdfs[j])
                elif mds_type == 'ks':
                    dissimilarity = test.ks(sample_cdfs[i], sample_cdfs[j])
                elif mds_type == 'kuiper':
                    dissimilarity = test.kuiper(sample_cdfs[i], sample_cdfs[j])
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

    fig.suptitle((title if title != "" else f"Multidimensional Scaling Plot") + f" (Stress: {np.round(stress, decimals=6)})")
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


def svg_to_png(decoded_svg):
    png_image_buffer = BytesIO()
    cairosvg.svg2png(bytestring=decoded_svg.encode('utf-8'), write_to=png_image_buffer)
    png_image_buffer.seek(0)
    png_data = png_image_buffer.getvalue()
    return png_data
