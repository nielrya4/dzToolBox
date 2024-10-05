import numpy as np
import plotly.graph_objects as go
import scipy.stats as st


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
        title=dict(text=title,
            font=dict(
                family=font,
                size=title_size,
                color='black'
            ),
        ),
        scene=dict(
            xaxis=dict(
                title=dict(
                    text=x_axis_title,
                    font=dict(
                        family=font,
                        size=font_size,
                        color='black'
                    )
                )
            ),
            yaxis=dict(
                title=dict(
                    text=y_axis_title,
                    font=dict(
                        family=font,
                        size=font_size,
                        color='black'
                    )
                )
            ),
            zaxis=dict(
                title=dict(
                    text=z_axis_title,
                    font=dict(
                        family=font,
                        size=font_size,
                        color='black'
                    )
                )
            ),
        ),
        width=fig_width*100,
        height=fig_height*100
    )
    html_str = fig.to_html(full_html=False)
    return html_str
