import numpy as np
import plotly.graph_objects as go
import scipy.stats as st

# Usage:
# x, y, z = kde_function_2d(sample)
# kde_html = kde_graph_2d(x, y, z)

def kde_function_2d(sample):
    x = [grain.age for grain in sample.grains]
    y = [grain.uncertainty for grain in sample.grains]

    # Define the borders
    deltaX = (max(x) - min(x)) / 10
    deltaY = (max(y) - min(y)) / 10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY

    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])

    # Gaussian KDE
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    # Also return the kernel function to evaluate points later
    return xx, yy, f, kernel


def kde_graph_2d(sample, title="2D Kernel Density Estimate", show_points=True):
    x, y, z, kernel = kde_function_2d(sample)

    # Create the surface plot
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])

    if show_points:
        # Get the coordinates of the points
        scatter_x = [grain.age for grain in sample.grains]
        scatter_y = [grain.uncertainty for grain in sample.grains]

        # Evaluate the KDE at the points' coordinates to get the intensity
        points = np.vstack([scatter_x, scatter_y])
        scatter_z = kernel(points)  # This returns the KDE intensity at each point

        # Add scatter plot of the points at their corresponding intensity
        scatter = go.Scatter3d(
            x=scatter_x,
            y=scatter_y,
            z=[z + 0.000003 for z in scatter_z],
            mode='markers',
            marker=dict(size=3, color='white', symbol='circle'),
            name='Data Points'
        )

        fig.add_trace(scatter)

    # Update the layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Age',
            yaxis_title='εHf(t)',
            zaxis_title='Intensity',
        ),
        width=800,
        height=800,
    )

    html_str = fig.to_html(full_html=False)
    return html_str
