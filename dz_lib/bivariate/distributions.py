import numpy as np
import plotly.graph_objects as go
import scipy.stats as st
from dz_lib.bivariate.data import BivariateSample
from dz_lib.utils import fonts
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib import font_manager


class BivariateDistribution:
    def __init__(
            self,
            mesh_x: [float],
            mesh_y: [float],
            mesh_z: [float],
            kernel: st.gaussian_kde,
            sample_x: [float],
            sample_y: [float]
    ):
        self.mesh_x = mesh_x
        self.mesh_y = mesh_y
        self.mesh_z = mesh_z
        self.kernel = kernel
        self.sample_x = sample_x
        self.sample_y = sample_y

def kde_function_2d(sample: BivariateSample):
    sample_x = [grain.age for grain in sample.grains]
    sample_y = [grain.hafnium for grain in sample.grains]
    deltaX = (max(sample_x) - min(sample_x)) / 10
    deltaY = (max(sample_y) - min(sample_y)) / 10
    xmin = min(sample_x) - deltaX
    xmax = max(sample_x) + deltaX
    ymin = min(sample_y) - deltaY
    ymax = max(sample_y) + deltaY
    mesh_x, mesh_y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([mesh_x.ravel(), mesh_y.ravel()])
    values = np.vstack([sample_x, sample_y])
    kernel = st.gaussian_kde(values)
    bandwidths = np.array([10, 0.25])
    kernel.covariance = np.diag(bandwidths ** 2)
    mesh_z = np.reshape(kernel(positions).T, mesh_x.shape)
    bivariate_distro = BivariateDistribution(mesh_x, mesh_y, mesh_z, kernel, sample_x, sample_y)
    return bivariate_distro


def kde_graph_2d(
        bivariate_distro: BivariateDistribution,
        title: str = None,
        show_points: bool = True,
        font_path: str=None,
        font_size: float = 12,
        fig_width: float = 9,
        fig_height: float = 7,
        x_axis_title: str = "Age (Ma)",
        y_axis_title: str = "εHf(t)",
        z_axis_title: str = "Intensity"
):
    font_files = font_manager.FontProperties(fname=font_path)
    font_name = font_files.get_name()
    mesh_x = bivariate_distro.mesh_x
    mesh_y = bivariate_distro.mesh_y
    mesh_z = bivariate_distro.mesh_z
    kernel = bivariate_distro.kernel
    sample_x = bivariate_distro.sample_x
    sample_y = bivariate_distro.sample_y
    title_size = font_size * 2
    fig = go.Figure(data=[go.Surface(z=mesh_z, x=mesh_x, y=mesh_y, colorscale='Viridis')])
    if show_points:
        points = np.vstack([sample_x, sample_y])
        scatter_z = kernel(points)
        scatter = go.Scatter3d(
            x=sample_x,
            y=sample_y,
            z=scatter_z,
            mode='markers',
            marker=dict(size=3, color='white', symbol='circle'),
            name='Data Points'
        )
        fig.add_trace(scatter)
    layout_dict = {
        "title": {
            "text": title,
            "font": {
                "family": font_name,
                "size": title_size,
                "color": "black"
            }
        },
        "scene": {
            "xaxis": {
                "title": {
                    "text": x_axis_title,
                    "font": {
                        "family": font_name,
                        "size": font_size,
                        "color": "black"
                    }
                }
            },
            "yaxis": {
                "title": {
                    "text": y_axis_title,
                    "font": {
                        "family": font_name,
                        "size": font_size,
                        "color": "black"
                    }
                }
            },
            "zaxis": {
                "title": {
                    "text": z_axis_title,
                    "font": {
                        "family": font_name,
                        "size": font_size,
                        "color": "black"
                    }
                }
            }
        },
        "width": fig_width * 100,
        "height": fig_height * 100
    }
    fig.update_layout(layout_dict)
    return fig

def heatmap(
        bivariate_distro: BivariateDistribution,
        show_points=False,
        font_path: str=None,
        font_size: float = 12,
        title="Heatmap",
        color_map="viridis",
        rescale_factor=1,
        fig_width=9,
        fig_height=7
):
    mesh_x = bivariate_distro.mesh_x
    mesh_y = bivariate_distro.mesh_y
    mesh_z = bivariate_distro.mesh_z
    sample_x = bivariate_distro.sample_x
    sample_y = bivariate_distro.sample_y
    x_rescaled = zoom(mesh_x, rescale_factor)
    y_rescaled = zoom(mesh_y, rescale_factor)
    z_rescaled = zoom(mesh_z, rescale_factor)
    if font_path:
        font = fonts.get_font(font_path)
    else:
        font = fonts.get_default_font()
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    title_size = font.get_size() * 2
    fig.suptitle(title, fontsize=title_size, fontproperties=font)
    c = ax.pcolormesh(x_rescaled, y_rescaled, z_rescaled, shading='gouraud', cmap=color_map, edgecolors='face')
    fig.colorbar(c, ax=ax)
    fig.text(0.5, 0.01, 'Age (Ma)', ha='center', va='center', fontsize=font_size, fontproperties=font)
    fig.text(0.01, 0.5, 'εHf(t)', va='center', rotation='vertical', fontsize=font_size, fontproperties=font)

    if show_points:
        ax.scatter(sample_x, sample_y, color='white', s=10, edgecolor='black', label='Data Points')
        ax.legend()
    plt.close()
    return fig