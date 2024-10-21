import numpy as np
import plotly.graph_objects as go
import scipy.stats as st
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from sqlalchemy import false


def kde_function_2d(sample):
    x = [grain.age for grain in sample.grains]
    y = [grain.uncertainty for grain in sample.grains]
    deltaX = (max(x) - min(x)) / 10
    deltaY = (max(y) - min(y)) / 10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    bandwidths = np.array([10, 0.25])
    kernel.covariance = np.diag(bandwidths ** 2)
    f = np.reshape(kernel(positions).T, xx.shape)
    return xx, yy, f, kernel

def kde_graph_2d(sample,
                 title="2D Kernel Density Estimate",
                 show_points=True,
                 font='ubuntu',
                 font_size=12,
                 fig_width=9,
                 fig_height=7,
                 x_axis_title="Age (Ma)",
                 y_axis_title="εHf(t)",
                 z_axis_title="Intensity"):
    title_size = font_size*2
    x, y, z, kernel = kde_function_2d(sample)
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])

    if show_points:
        scatter_x = [grain.age for grain in sample.grains]
        scatter_y = [grain.uncertainty for grain in sample.grains]
        points = np.vstack([scatter_x, scatter_y])
        scatter_z = kernel(points)
        scatter = go.Scatter3d(
            x=scatter_x,
            y=scatter_y,
            z=scatter_z,
            mode='markers',
            marker=dict(size=3, color='white', symbol='circle'),
            name='Data Points'
        )
        fig.add_trace(scatter)

    fig.update_layout(
        width=fig_width * 100,
        height=fig_height * 100,
        title={
            "text": title,
            "font": {
                "family": font,
                "size": title_size,
                "color": "black"
            },
        },
        scene={
            "xaxis": {
                "title": {
                    "text": x_axis_title,
                    "font": {
                        "family": font,
                        "size": font_size,
                        "color": "black"
                    }
                }
            },
            "yaxis": {
                "title": {
                    "text": y_axis_title,
                    "font": {
                        "family": font,
                        "size": font_size,
                        "color": "black"
                    }
                }
            },
            "zaxis": {
                "title": {
                    "text": z_axis_title,
                    "font": {
                        "family": font,
                        "size": font_size,
                        "color": "black"
                    }
                }
            }
        }
    )
    config = {
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'multivariate_plot',
            'height': fig_height*100,
            'width': fig_width*100,
            'scale': 1
        }
    }
    html_str = fig.to_html(full_html=False, config=config)
    return html_str


def heatmap(x, y, z, title="Heatmap", color_map="viridis", rescale_factor=1, fig_width=9, fig_height=7):
    x_rescaled = zoom(x, rescale_factor)
    y_rescaled = zoom(y, rescale_factor)
    z_rescaled = zoom(z, rescale_factor)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    c = ax.pcolormesh(x_rescaled, y_rescaled, z_rescaled, shading='gouraud', cmap=color_map, edgecolors='face')
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('Age (Ma)')
    ax.set_ylabel('εHf(t)')
    ax.set_title(title)
    return fig