import matplotlib.pyplot as plt
import  matplotlib.gridspec as gridspec
from dz_lib.univariate.data import Sample
from dz_lib.utils import fonts, encode
import numpy as np

class BinList:
    """
    Defines a non-overlapping age bin scheme.

    Bins are interpreted as [min, max) intervals.
    """

    def __init__(
        self,
        edges,
        color_map: str = 'plasma',
        labels: list[str] | None = None
    ):
        edges = sorted(edges)
        if len(edges) < 2:
            raise ValueError("BinList requires at least 2 edges (1 bin).")
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                raise ValueError(
                    f"Edges must be strictly increasing, got {edges[i - 1]} and {edges[i]}."
                )

        self.bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        self.n_bins = len(self.bins)

        # Labels
        if labels is not None:
            if len(labels) != self.n_bins:
                raise ValueError("Number of labels must match number of bins.")
            self.labels = labels
        else:
            self.labels = [
                f'{int(lo)}–{int(hi)}'
                for lo, hi in self.bins
            ]

        # Colors
        cmap = plt.cm.get_cmap(color_map, self.n_bins)
        self.colors = cmap(np.linspace(0, 1, self.n_bins))

    def count(self, ages: np.ndarray) -> np.ndarray:
        """
        Count values into bins.
        """
        return np.array([
            np.sum((ages >= lo) & (ages < hi))
            for lo, hi in self.bins
        ])


def histogram_graph(
    samples: list,
    bin_list: BinList,
    legend: bool = True,
    title: str = None,
    font_path: str = None,
    font_size: float = 12,
    fig_width: float = 9,
    fig_height: float = 7,
    color_map: str = 'plasma',
    fill: bool = True,
):
    num_samples = len(samples)

    # Colormap
    colors = plt.cm.get_cmap(color_map, num_samples)(
        np.linspace(0, 1, num_samples)
    )

    # Fonts
    if font_path:
        font = fonts.get_font(font_path)
    else:
        font = fonts.get_default_font()

    # Histogram bins — extract edges from BinList, handling potential gaps
    bin_edges = []
    for lo, hi in bin_list.bins:
        if not bin_edges or bin_edges[-1] != lo:
            bin_edges.append(lo)
        bin_edges.append(hi)
    x_min, x_max = bin_edges[0], bin_edges[-1]

    # Always stack if more than one sample
    stacked = num_samples > 1

    if not stacked:
        fig, ax = plt.subplots(
            figsize=(fig_width, fig_height),
            dpi=100,
            squeeze=False
        )
        ax_list = [ax[0, 0]]

        sample = samples[0]
        ages = [g.age for g in sample.grains]

        ax[0, 0].hist(
            ages,
            bins=bin_edges,
            color=colors[0],
            alpha=0.75 if fill else 1.0,
            edgecolor='black',
            linewidth=1,
            label=sample.name
        )

        if legend:
            ax[0, 0].legend(
                loc='upper left',
                bbox_to_anchor=(1, 1),
                fontsize=font_size
            )

    else:
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
        gs = gridspec.GridSpec(num_samples, 1, figure=fig)
        ax_list = []

        for i, sample in enumerate(samples):
            ax = fig.add_subplot(gs[i])
            ax_list.append(ax)

            ages = [g.age for g in sample.grains]

            ax.hist(
                ages,
                bins=bin_edges,
                color=colors[i],
                alpha=0.75 if fill else 1.0,
                edgecolor='black',
                linewidth=1,
                label=sample.name
            )

            if legend:
                ax.legend(
                    loc='upper left',
                    bbox_to_anchor=(1, 1),
                    fontsize=font_size
                )

    # Shared axis styling
    for i, ax in enumerate(ax_list):
        ax.set_xlim(x_min, x_max)
        ax.tick_params(axis='both', which='major', labelsize=font_size)

        if stacked and i < len(ax_list) - 1:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            plt.setp(ax.get_xticklabels(), fontproperties=font)

        plt.setp(ax.get_yticklabels(), fontproperties=font)

        ax.set_facecolor('white')
        ax.tick_params(axis='x', colors='black', width=2)
        ax.tick_params(axis='y', colors='black', width=2)

    # Titles and labels
    fig.suptitle(
        title,
        fontsize=font_size * 1.75,
        fontproperties=font
    )
    fig.text(
        0.5, 0.02, 'Age (Ma)',
        ha='center',
        va='center',
        fontsize=font_size,
        fontproperties=font
    )
    fig.text(
        0.01, 0.5, 'Count',
        va='center',
        rotation='vertical',
        fontsize=font_size,
        fontproperties=font
    )

    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.96])
    plt.xlim(x_min, x_max)

    plt.close()
    return fig

def histogram_pie_chart(
    samples: list,
    bin_list: BinList,
    title: str = None,
    font_path: str = None,
    font_size: float = 12,
    fig_width: float = 12,
    n_cols: int = 2,  # number of columns
    min_label_pct: float = 5,  # minimum percentage to show wedge label
):
    num_samples = len(samples)

    # Fonts
    if font_path:
        font = fonts.get_font(font_path)
    else:
        font = fonts.get_default_font()

    # Calculate number of rows and figure height
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig_height = max(5, n_rows * 3.5 + 1.5)  # 3.5 inches per pie row + 1.5 for legend strip

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
    gs = gridspec.GridSpec(
        n_rows + 1, n_cols, figure=fig,
        height_ratios=[3.5] * n_rows + [1],
        hspace=0.6,  # vertical spacing
        wspace=0.4   # horizontal spacing
    )

    # Legend strip: colored rectangles spanning all columns in last row
    ax_legend = fig.add_subplot(gs[n_rows, :])
    xranges = [(lo, hi - lo) for lo, hi in bin_list.bins]
    ax_legend.broken_barh(xranges, (0, 1), facecolors=bin_list.colors, edgecolors='black', linewidth=1)
    for (lo, hi), label, color in zip(bin_list.bins, bin_list.labels, bin_list.colors):
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = 'black' if luminance > 0.5 else 'white'
        ax_legend.text(
            (lo + hi) / 2, 0.5, label,
            ha='center', va='center',
            fontsize=font_size * 0.65,
            fontproperties=font,
            color=text_color
        )
    ax_legend.set_xlim(bin_list.bins[0][0], bin_list.bins[-1][1])
    ax_legend.set_ylim(0, 1)
    ax_legend.set_yticks([])
    ax_legend.tick_params(axis='x', labelsize=font_size * 0.8)
    plt.setp(ax_legend.get_xticklabels(), fontproperties=font)
    ax_legend.set_xlabel('Age (Ma)', fontsize=font_size * 0.85, fontproperties=font, labelpad=15)
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.spines['right'].set_visible(False)

    axes_samples = []

    for i, sample in enumerate(samples):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        axes_samples.append((ax, sample))

    # Hide unused axes if grid is larger than sample count
    for j in range(num_samples, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        fig.add_subplot(gs[row, col]).set_visible(False)

    # Annotation helper
    def annotate_pie(ax, wedges, counts):
        total = counts.sum()
        for wedge, count in zip(wedges, counts):
            pct = 100 * count / total
            if pct < min_label_pct:
                continue
            angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
            x, y = np.cos(angle), np.sin(angle)
            lx, ly = 1.3 * x, 1.3 * y
            ha = 'left' if lx > 0 else 'right'
            ax.annotate(
                f'{pct:.1f}%',
                xy=(x, y),
                xytext=(lx, ly),
                ha=ha,
                va='center',
                fontsize=font_size * 0.7,
                arrowprops=dict(
                    arrowstyle='-',
                    color='gray',
                    linewidth=1
                )
            )

    # Draw pies
    for ax, sample in axes_samples:
        ages = np.array([g.age for g in sample.grains])
        counts = bin_list.count(ages)

        nonzero = counts > 0
        plot_counts = counts[nonzero]
        plot_colors = bin_list.colors[nonzero]

        ax.set_title(
            sample.name,
            fontsize=font_size * 0.9,
            fontproperties=font,
            pad=15  # increase vertical space above the pie
        )
        wedges, _ = ax.pie(
            plot_counts,
            colors=plot_colors,
            startangle=90,
            radius=0.8
        )
        ax.set_aspect('equal')

        annotate_pie(ax, wedges, plot_counts)



    if title:
        fig.suptitle(
            title,
            fontsize=font_size * 1.5,
            fontproperties=font
        )

    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.97])

    plt.close()
    return fig

