from dz_lib.univariate.data import Sample
from dz_lib.utils import fonts, encode
from dz_lib import config
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Distribution:
    def __init__(self, name, x_values, y_values):
        self.name = name
        self.x_values = x_values
        self.y_values = y_values

    def subset(self, x_min: float, x_max: float):
        mask = (self.x_values > x_min) & (self.x_values < x_max)
        new_y_vals = np.where(mask, self.y_values, 0)
        return Distribution(self.name, self.x_values, new_y_vals)


# ----------------------------------------------------------------------
# Peak detection (guaranteed, KDE-safe)
# ----------------------------------------------------------------------
def find_peaks(x: np.ndarray, y: np.ndarray, n_peaks: int):
    """
    Find up to n_peaks peaks in a smooth distribution.
    Always includes the global maximum.
    """
    if n_peaks <= 0:
        return []

    peaks = []

    max_idx = np.argmax(y)
    peaks.append((x[max_idx], y[max_idx]))

    if n_peaks == 1:
        return peaks

    dy = np.diff(y)
    signs = np.sign(dy)

    candidates = np.where(
        (signs[:-1] >= 0) & (signs[1:] <= 0)
    )[0] + 1

    candidates = [i for i in candidates if i != max_idx]
    candidates = sorted(candidates, key=lambda i: y[i], reverse=True)

    for i in candidates:
        peaks.append((x[i], y[i]))
        if len(peaks) >= n_peaks:
            break

    return peaks


def find_modes(x: np.ndarray, y: np.ndarray, n_modes: int):
    """
    Robust mode finder for smooth KDE/PDP curves.
    Always includes the global maximum.
    """
    modes = [(x[np.argmax(y)], np.max(y))]

    if n_modes == 1:
        return modes

    dy = np.diff(y)
    signs = np.sign(dy)

    candidates = np.where(
        (signs[:-1] >= 0) & (signs[1:] <= 0)
    )[0] + 1

    candidates = [i for i in candidates if i != np.argmax(y)]
    candidates = sorted(candidates, key=lambda i: y[i], reverse=True)

    for i in candidates:
        modes.append((x[i], y[i]))
        if len(modes) >= n_modes:
            break

    return modes


# ----------------------------------------------------------------------
# Distribution generators
# ----------------------------------------------------------------------
def kde_function(
    sample: Sample,
    bandwidth: float = 10,
    x_min: float = 0,
    x_max: float = 4500,
):
    n_steps = int(x_max - x_min + 1)
    x_values = np.linspace(x_min, x_max, n_steps)

    ages = np.array([grain.age for grain in sample.grains])

    ages_2d = ages[:, np.newaxis]
    x_2d = x_values[np.newaxis, :]

    diff_squared = (x_2d - ages_2d) ** 2
    variance_2 = 2 * bandwidth ** 2
    normalization = 1.0 / (np.sqrt(2 * np.pi) * bandwidth)

    kernels = normalization * np.exp(-diff_squared / variance_2)
    y_values = np.sum(kernels, axis=0)

    y_values /= np.sum(y_values)
    return Distribution(sample.name, x_values, y_values)


def pdp_function(
    sample: Sample,
    x_min: float = 0,
    x_max: float = 4500,
    sigma: Optional[int] = None
):
    """
    Compute Probability Density Plot (PDP) for a sample.

    Parameters
    ----------
    sample : Sample
        The sample containing grains with ages and uncertainties
    x_min : float
        Minimum x value (age) for the distribution
    x_max : float
        Maximum x value (age) for the distribution
    sigma : int, optional
        Sigma level to use for bandwidths. If None, uses config.sigma_in
        (i.e., the raw input uncertainties). Use 1 for 1-sigma or 2 for 2-sigma.
    """
    n_steps = int(x_max - x_min + 1)
    x_values = np.linspace(x_min, x_max, n_steps)

    ages = np.array([grain.age for grain in sample.grains])

    # Get bandwidths at specified sigma level
    if sigma is None:
        # Use raw uncertainties (at whatever sigma they were provided)
        bandwidths = np.array([grain.uncertainty for grain in sample.grains])
    else:
        # Convert to requested sigma level
        bandwidths = np.array([grain.get_uncertainty(sigma) for grain in sample.grains])

    ages_2d = ages[:, np.newaxis]
    bandwidths_2d = bandwidths[:, np.newaxis]
    x_2d = x_values[np.newaxis, :]

    diff_squared = (x_2d - ages_2d) ** 2
    variance_2 = 2 * bandwidths_2d ** 2
    normalization = 1.0 / (np.sqrt(2 * np.pi) * bandwidths_2d)

    kernels = normalization * np.exp(-diff_squared / variance_2)
    y_values = np.sum(kernels, axis=0)

    y_values /= np.sum(y_values)
    return Distribution(sample.name, x_values, y_values)


def cdf_function(distribution: Distribution):
    cdf = np.cumsum(distribution.y_values)
    cdf /= cdf[-1]
    return Distribution(distribution.name, distribution.x_values, cdf)


# ----------------------------------------------------------------------
# Smart label positioning with collision detection
# ----------------------------------------------------------------------
def _position_mode_labels(modes, x_values, y_values, x_range, font_size):
    """
    Position mode labels on top of peaks, moving them to avoid collisions.
    Default: directly above peak (no line needed)
    If collision: move to the right (connection line will be added)

    Args:
        modes: List of (x, y) tuples representing mode positions
        x_values: X-axis values of the distribution curve
        y_values: Y-axis values of the distribution curve
        x_range: (x_min, x_max) tuple for the plot range
        font_size: Font size for labels

    Returns:
        List of (mode_x, mode_y, x_offset, y_offset, needs_line) tuples
    """
    if not modes:
        return []

    # Sort modes by x position (left to right)
    sorted_modes = sorted(modes, key=lambda m: m[0])

    x_min, x_max = x_range
    plot_width = x_max - x_min
    y_max = np.max(y_values)
    y_min = np.min(y_values)
    y_range = y_max - y_min

    # Estimate label width in data coordinates
    # For a 4-digit number at typical font size
    # Scale with font size - add margin so labels don't appear too close
    font_scale = font_size / 12.0  # Relative to default font size
    label_width_data = plot_width * 0.045 * font_scale  # With comfortable margin
    label_height_data = y_range * 0.10 * font_scale  # With comfortable margin

    positions = []
    occupied_regions = []  # Track (x_left, x_right, y_bottom, y_top)
    label_directions = []  # Track which direction each label went: 'center', 'left', or 'right'

    for mode_idx, (px, py) in enumerate(sorted_modes):
        # Default position: directly above peak
        default_y_offset_points = 4
        # Convert y offset to data coordinates more accurately
        default_y_offset_data = y_range * 0.05  # 5% of range above peak

        # Check for collisions at default position
        label_x = px
        label_y = py + default_y_offset_data

        # Label bounding box at default position
        label_left = label_x - label_width_data / 2
        label_right = label_x + label_width_data / 2
        label_bottom = label_y - label_height_data / 2
        label_top = label_y + label_height_data / 2

        has_collision = False

        # Check collision with other labels
        for (ox_left, ox_right, oy_bottom, oy_top) in occupied_regions:
            # Check if rectangles overlap
            if not (label_right < ox_left or label_left > ox_right or
                    label_top < oy_bottom or label_bottom > oy_top):
                has_collision = True
                break

        # Check collision with distribution curve
        if not has_collision:
            # Sample points across the label width
            n_samples = 10
            for i in range(n_samples):
                check_x = label_left + (i / (n_samples - 1)) * (label_right - label_left)
                if check_x < x_min or check_x > x_max:
                    continue  # Skip out of bounds, don't flag as collision

                # Find curve value at this x position
                idx = np.argmin(np.abs(x_values - check_x))
                curve_y = y_values[idx]

                # Check if curve intersects label bounding box
                # Only actual intersection counts, no extra margin
                if curve_y >= label_bottom:
                    has_collision = True
                    break

        # If no collision, use default position (no line needed)
        if not has_collision:
            x_offset = 0
            y_offset = default_y_offset_points
            needs_line = False
            chosen_direction = 'center'
        else:
            # Collision detected - try positions to the right and left
            found_position = False

            # Determine preferred direction to avoid crossing lines
            # Priority 1: Leftmost peak prefers left, rightmost prefers right
            # Priority 2: Alternate with previous label to avoid crossing
            # Priority 3: Check which side has less occupation
            prefer_left = False

            # Check if this is the leftmost or rightmost peak
            is_leftmost = (mode_idx == 0)
            is_rightmost = (mode_idx == len(sorted_modes) - 1)

            if is_leftmost:
                # Leftmost peak: try left first to spread outward
                prefer_left = True
            elif is_rightmost:
                # Rightmost peak: try right first to spread outward
                prefer_left = False
            elif mode_idx > 0 and len(label_directions) > 0:
                # Middle peaks: alternate with previous label to avoid crossing
                prev_direction = label_directions[-1]
                if prev_direction == 'right':
                    prefer_left = True
                elif prev_direction == 'left':
                    prefer_left = False
                # If previous was 'center', fall through to occupation check
                else:
                    # Check occupation as fallback
                    left_occupied = 0
                    right_occupied = 0
                    for (ox_left, ox_right, oy_bottom, oy_top) in occupied_regions:
                        if ox_right < px:
                            left_occupied += 1
                        elif ox_left > px:
                            right_occupied += 1
                    prefer_left = left_occupied < right_occupied
            else:
                # Fallback: check occupation
                left_occupied = 0
                right_occupied = 0
                for (ox_left, ox_right, oy_bottom, oy_top) in occupied_regions:
                    if ox_right < px:
                        left_occupied += 1
                    elif ox_left > px:
                        right_occupied += 1
                prefer_left = left_occupied < right_occupied

            offset_multipliers = [1.2, 2.0, 3.0, 4.5, 6.5]
            candidate_offsets = []

            # Build candidate list with preferred direction first
            for mult in offset_multipliers:
                if prefer_left:
                    candidate_offsets.append(('left', label_width_data * mult))
                    candidate_offsets.append(('right', label_width_data * mult))
                else:
                    candidate_offsets.append(('right', label_width_data * mult))
                    candidate_offsets.append(('left', label_width_data * mult))

            for direction, offset_data in candidate_offsets:
                if direction == 'right':
                    test_x = px + offset_data
                else:  # left
                    test_x = px - offset_data

                test_y = label_y

                # Test label bounding box
                test_left = test_x - label_width_data / 2
                test_right = test_x + label_width_data / 2
                test_bottom = test_y - label_height_data / 2
                test_top = test_y + label_height_data / 2

                # Check if out of bounds
                if test_right > x_max or test_left < x_min:
                    continue

                clear = True

                # Check against other labels
                for (ox_left, ox_right, oy_bottom, oy_top) in occupied_regions:
                    if not (test_right < ox_left or test_left > ox_right or
                            test_top < oy_bottom or test_bottom > oy_top):
                        clear = False
                        break

                # Check against curve
                if clear:
                    for i in range(n_samples):
                        check_x = test_left + (i / (n_samples - 1)) * (test_right - test_left)
                        if check_x < x_min or check_x > x_max:
                            continue  # Skip out of bounds

                        idx = np.argmin(np.abs(x_values - check_x))
                        curve_y = y_values[idx]

                        # Only actual intersection counts
                        if curve_y >= test_bottom:
                            clear = False
                            break

                if clear:
                    # Convert data offset back to points for annotation
                    if direction == 'right':
                        x_offset = int((offset_data / plot_width) * 500)  # positive for right
                        chosen_direction = 'right'
                    else:
                        x_offset = -int((offset_data / plot_width) * 500)  # negative for left
                        chosen_direction = 'left'

                    y_offset = default_y_offset_points
                    needs_line = True
                    found_position = True
                    label_x = test_x
                    label_left = test_left
                    label_right = test_right
                    label_bottom = test_bottom
                    label_top = test_top
                    break

            # Fallback: if no clear position found, stack vertically
            if not found_position:
                # Move up above other labels with extra spacing
                max_y = py + default_y_offset_data  # Start from default position
                for (ox_left, ox_right, oy_bottom, oy_top) in occupied_regions:
                    # If horizontally overlapping with this peak
                    if not (label_right < ox_left or label_left > ox_right):
                        max_y = max(max_y, oy_top)

                # Place above the highest overlapping label with spacing
                label_y = max_y + label_height_data * 0.6  # Add 60% of height as spacing
                label_bottom = label_y - label_height_data / 2
                label_top = label_y + label_height_data / 2

                # Convert to points offset
                # More accurate conversion based on actual data coordinate difference
                y_offset_data = label_y - py
                # Rough estimate: assume figure is ~500 points tall, y_range maps to that
                y_offset = int((y_offset_data / y_range) * 500)
                x_offset = 0
                needs_line = True
                chosen_direction = 'center'  # Vertical stack is centered

        # Record this occupied region and direction
        occupied_regions.append((label_left, label_right, label_bottom, label_top))
        label_directions.append(chosen_direction)
        positions.append((px, py, x_offset, y_offset, needs_line))

    return positions


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def distribution_graph(
    distributions: list,
    x_min: float = 0,
    x_max: float = 4500,
    stacked: bool = False,
    legend: bool = True,
    title: str = None,
    font_path: str = None,
    font_size: float = 12,
    fig_width: float = 9,
    fig_height: float = 7,
    color_map: str = 'plasma',
    modes_labeled: int = 0,
    fill: bool = False,
    mode_label_lines: bool = True,
):
    num_samples = len(distributions)
    colors = plt.cm.get_cmap(color_map, num_samples)(
        np.linspace(0, 1, num_samples)
    )

    if font_path:
        font = fonts.get_font(font_path)
    else:
        font = fonts.get_default_font()

    if not stacked:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100, squeeze=False)
        ax_list = [ax[0, 0]]

        # Track max y value to extend y-axis for labels
        global_y_max = 0

        for i, dist in enumerate(distributions):
            x, y = dist.x_values, dist.y_values
            global_y_max = max(global_y_max, np.max(y))

            ax[0, 0].plot(x, y, label=dist.name, color=colors[i])

            if fill:
                ax[0, 0].fill_between(
                    x, y, 0,
                    color=colors[i],
                    alpha=0.25,
                    linewidth=0
                )

            if modes_labeled > 0:
                modes = find_modes(x, y, modes_labeled)
                positions = _position_mode_labels(modes, x, y, (x_min, x_max), font_size)

                for (px, py, x_offset, y_offset, needs_line) in positions:
                    # Add connection line only if label was moved due to collision
                    arrow_props = None
                    if needs_line and mode_label_lines:
                        arrow_props = dict(
                            arrowstyle='-',
                            color='gray',
                            alpha=0.6,
                            linewidth=2,
                            linestyle='--'
                        )

                    ax[0, 0].annotate(
                        f'{px:.0f}',
                        xy=(px, py),
                        xytext=(x_offset, y_offset),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=font_size * 0.85,
                        zorder=5,
                        clip_on=True,
                        arrowprops=arrow_props
                    )

        # Extend y-axis to accommodate labels
        # Calculate headroom dynamically based on font size
        if modes_labeled > 0:
            # Estimate: font_size points + offset (4 points) + margin (20 points)
            # Convert points to fraction of figure: points / (fig_height_inches * 72)
            label_space_points = font_size * 0.85 + 4 + 20  # label font + offset + margin
            label_space_fraction = label_space_points / (fig_height * 72)
            # Add this fraction to the y-range
            headroom_multiplier = 1.0 + label_space_fraction
            ax[0, 0].set_ylim(bottom=0, top=global_y_max * headroom_multiplier)

        if legend:
            ax[0, 0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)

    else:
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
        gs = gridspec.GridSpec(len(distributions), 1, figure=fig)
        ax_list = []

        for i, dist in enumerate(distributions):
            ax = fig.add_subplot(gs[i])
            ax_list.append(ax)

            x, y = dist.x_values, dist.y_values
            y_max = np.max(y)

            ax.plot(x, y, label=dist.name, color=colors[i])

            if fill:
                ax.fill_between(
                    x, y, 0,
                    color=colors[i],
                    alpha=0.25,
                    linewidth=0
                )

            if modes_labeled > 0:
                modes = find_modes(x, y, modes_labeled)
                positions = _position_mode_labels(modes, x, y, (x_min, x_max), font_size)

                for (px, py, x_offset, y_offset, needs_line) in positions:
                    # Add connection line only if label was moved due to collision
                    arrow_props = None
                    if needs_line and mode_label_lines:
                        arrow_props = dict(
                            arrowstyle='-',
                            color='gray',
                            alpha=0.6,
                            linewidth=2,
                            linestyle='--'
                        )

                    ax.annotate(
                        f'{px:.0f}',
                        xy=(px, py),
                        xytext=(x_offset, y_offset),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=font_size * 0.85,
                        zorder=5,
                        clip_on=True,
                        arrowprops=arrow_props
                    )

                # Extend y-axis to accommodate labels
                # Calculate headroom dynamically based on font size
                # For stacked plots, each subplot gets less height, so adjust accordingly
                subplot_height = fig_height / len(distributions)
                label_space_points = font_size * 0.85 + 4 + 20  # label font + offset + margin
                label_space_fraction = label_space_points / (subplot_height * 72)
                headroom_multiplier = 1.0 + label_space_fraction
                ax.set_ylim(bottom=0, top=y_max * headroom_multiplier)

            if legend:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)

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

    fig.suptitle(title, fontsize=font_size * 1.75, fontproperties=font)
    fig.text(0.5, 0.02, 'Age (Ma)', ha='center', va='center',
             fontsize=font_size, fontproperties=font)
    fig.text(0.01, 0.5, 'Probability Differential', va='center',
             rotation='vertical', fontsize=font_size, fontproperties=font)

    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.96])
    plt.xlim(x_min, x_max)

    plt.close()
    return fig
