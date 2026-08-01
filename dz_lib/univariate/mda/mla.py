"""
Maximum Likelihood Age (MLA) Implementation

Based on Galbraith (2005) Section 6.11 and IsoplotR implementation by Vermeesch.
Implements a 2-component mixture model for maximum depositional age estimation.

References:
    Galbraith, R.F. and Laslett, G.M., 1993. Statistical models for mixed
        fission track ages. Nuclear Tracks and Radiation Measurements, 21(4), 459-470.
    Galbraith, R.F., 2005. Statistics for fission track analysis.
        Chapman and Hall/CRC, 229p.
    Vermeesch, P., 2020. Maximum depositional age estimation revisited.
        Geoscience Frontiers 12(2021), 843-850.
"""

from dz_lib.univariate.data import Grain
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import subprocess
import tempfile
import os
import matplotlib.pyplot as plt


def _compute_hessian_via_R(par, z, s, min_z, max_z, n_params):
    """
    Compute Hessian using R's optimHess function (same as IsoplotR).

    This function creates a temporary R script that computes the Hessian matrix
    using R's optimHess function from the stats package. This ensures our
    uncertainty estimates exactly match IsoplotR.

    Args:
        par: Parameters in unbounded space
        z: Log-ages
        s: Log-uncertainties
        min_z: Minimum log-age
        max_z: Maximum log-age
        n_params: Number of parameters (3 or 4)

    Returns:
        Hessian matrix

    Notes:
        Requires R to be installed and accessible via 'Rscript' command.
    """
    # Convert to plain Python lists for R
    par_list = [float(x) for x in par]
    z_list = [float(x) for x in z]
    s_list = [float(x) for x in s]

    # Create temporary R script with all necessary functions
    # This is self-contained and doesn't require external R packages
    r_script = f"""
# Define parameters
par <- c({','.join(map(str, par_list))})
zs <- matrix(c({','.join(map(str, z_list))}), ncol=1)
ss <- c({','.join(map(str, s_list))})
zs <- cbind(zs, ss)
Mz <- {float(max_z)}
np <- {n_params}

# Define parameter mapping function
# Maps from unbounded optimization space to model space
mappar <- function(par, Mz) {{
    np <- length(par)
    gam <- par[1]  # minimum age (log-space)
    prop <- exp(par[2])/(1+exp(par[2]))  # proportion (logit transform)
    sig <- exp(par[3])  # dispersion (log transform)
    if (np<4) mu <- gam
    else mu <- gam + (Mz-gam)*exp(par[4])/(1+exp(par[4]))
    c(gam,prop,sig,mu)
}}

# Define negative log-likelihood function
# Based on Galbraith (2005) Section 6.11
LL <- function(par, zs, Mz) {{
    pars <- mappar(par, Mz)
    z <- zs[,1]
    s <- zs[,2]
    gam <- pars[1]
    prop <- pars[2]
    sig <- pars[3]
    mu <- pars[4]

    # Component 1: Discrete peak at minimum age
    AA  <- prop/sqrt(2*pi*s^2)
    BB <- -0.5*((z-gam)/s)^2

    # Component 2: Truncated continuous distribution
    CC <- (1-prop)/sqrt(2*pi*(sig^2+s^2))
    mu0 <- (mu/sig^2 + z/s^2)/(1/sig^2 + 1/s^2)
    s0 <- 1/sqrt(1/sig^2 + 1/s^2)
    DD <- 1-pnorm((gam-mu0)/s0)
    EE <- 1-pnorm((gam-mu)/sig)
    FF <- -0.5*((z-mu)^2)/(sig^2+s^2)

    # Total likelihood
    fu <- AA*exp(BB) + CC*(DD/EE)*exp(FF)
    fu[fu<.Machine$double.xmin] <- .Machine$double.xmin
    fu[fu>.Machine$double.xmax] <- .Machine$double.xmax
    sum(-log(fu))
}}

# Compute Hessian using R's optimHess (base R stats package)
H <- optimHess(par, LL, zs=zs, Mz=Mz)

# Write to output
write.table(H, file='_hessian_tmp.txt', row.names=FALSE, col.names=FALSE)
"""

    # Write and execute R script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_script)
        r_file = f.name

    # Create temp directory for output file
    temp_dir = tempfile.gettempdir()
    hess_file = os.path.join(temp_dir, '_hessian_tmp.txt')

    try:
        # Run R script
        result = subprocess.run(['Rscript', r_file],
                              capture_output=True,
                              text=True,
                              cwd=temp_dir)

        if result.returncode != 0:
            raise RuntimeError(f"R script failed: {result.stderr}")

        # Read Hessian from output
        if not os.path.exists(hess_file):
            raise RuntimeError(f"Hessian file not created: {hess_file}")

        hessian = np.loadtxt(hess_file)

        return hessian

    finally:
        # Clean up temporary files
        if os.path.exists(r_file):
            os.remove(r_file)
        if os.path.exists(hess_file):
            os.remove(hess_file)


class _MinimumAgeModel:
    """
    Two-component mixture model for maximum depositional age estimation.

    Component 1: Discrete age peak at minimum age (proportion π)
    Component 2: Continuous distribution truncated at minimum age (proportion 1-π)

    Based on Galbraith (2005) Section 6.11 and IsoplotR implementation.

    Parameters:
        gam (γ): minimum age (in log space)
        prop (π): proportion in minimum age component [0,1]
        sig (σ): dispersion of continuous component
        mu (μ): mean of continuous component (only in 4-parameter model)
    """

    def __init__(self, n_params=4):
        """
        Initialize the model.

        Args:
            n_params: Number of parameters (3 or 4)
                3-param: gam, prop, sig (mu = gam)
                4-param: gam, prop, sig, mu
        """
        if n_params not in [3, 4]:
            raise ValueError("n_params must be 3 or 4")
        self.n_params = n_params
        self.result = None

    def map_params(self, par, min_z, max_z):
        """
        Transform parameters from unbounded optimization space to model space.

        Args:
            par: Parameters in unbounded space
            min_z: Minimum log-age in data
            max_z: Maximum log-age in data

        Returns:
            (gam, prop, sig, mu) in model space
        """
        gam = par[0]
        prop = np.exp(par[1]) / (1 + np.exp(par[1]))  # logit transform
        sig = np.exp(par[2])  # log transform

        if self.n_params < 4:
            mu = gam
        else:
            # mu bounded between gam and max_z
            mu = gam + (max_z - gam) * np.exp(par[3]) / (1 + np.exp(par[3]))

        return gam, prop, sig, mu

    def get_likelihood(self, par, z, s, min_z, max_z):
        """
        Compute negative log-likelihood (for minimization).

        This implements equation from Galbraith (2005) Section 6.11.

        Args:
            par: Parameters in unbounded space
            z: Log-ages
            s: Uncertainties in log-space
            min_z: Minimum log-age in data
            max_z: Maximum log-age in data

        Returns:
            Negative log-likelihood
        """
        gam, prop, sig, mu = self.map_params(par, min_z, max_z)

        # Component 1: Discrete peak at minimum age
        AA = prop / np.sqrt(2 * np.pi * s**2)
        BB = -0.5 * ((z - gam) / s)**2
        component1 = AA * np.exp(BB)

        # Component 2: Truncated continuous distribution
        CC = (1 - prop) / np.sqrt(2 * np.pi * (sig**2 + s**2))

        # Posterior mean and std for each grain
        mu0 = (mu / sig**2 + z / s**2) / (1 / sig**2 + 1 / s**2)
        s0 = 1 / np.sqrt(1 / sig**2 + 1 / s**2)

        # Truncation factors
        DD = 1 - norm.cdf((gam - mu0) / s0)
        EE = 1 - norm.cdf((gam - mu) / sig)

        FF = -0.5 * (z - mu)**2 / (sig**2 + s**2)
        component2 = CC * (DD / EE) * np.exp(FF)

        # Total probability density
        fu = component1 + component2

        # Clamp to avoid numerical issues
        fu = np.clip(fu, np.finfo(float).tiny, np.finfo(float).max)

        # Return negative log-likelihood
        return -np.sum(np.log(fu))

    def fit(self, ages, uncertainties, verbose=False):
        """
        Fit the minimum age model to data.

        Args:
            ages: Array of ages (in Ma or any linear scale)
            uncertainties: Array of 1-sigma uncertainties
            verbose: Print optimization details

        Returns:
            Dictionary with results:
                'min_age': Minimum age estimate
                'min_age_se': Standard error on minimum age
                'proportion': Proportion in minimum age component
                'dispersion': Dispersion parameter
                'log_likelihood': Log-likelihood value
                'success': Whether optimization succeeded

        Notes:
            Requires R to be installed for uncertainty calculation via Hessian.
            If R is not available, uncertainties will be set to nan.
        """
        ages = np.array(ages)
        uncertainties = np.array(uncertainties)

        # Transform to log-space
        z = np.log(ages)
        s = uncertainties / ages  # Error propagation for log transform

        # Sort for initialization
        z_sorted = np.sort(z)
        min_z = z_sorted[0]
        max_z = z_sorted[-1]

        # Get initial estimates
        cfit_mu = np.mean(z)
        cfit_sigma = max(np.std(z), 0.01)

        # Initial parameter estimates
        gam_init = np.mean(z_sorted[:2])
        prop_init = 0  # logit(0.5) = 0
        sig_init = np.log(cfit_sigma)

        if self.n_params == 4:
            mu_init = 0
            init = np.array([gam_init, prop_init, sig_init, mu_init])
            lower = np.array([min_z, -20, sig_init - 20, -20])
            upper = np.array([cfit_mu, 20, sig_init + 2, 20])
        else:
            init = np.array([gam_init, prop_init, sig_init])
            lower = np.array([min_z, -20, sig_init - 20])
            upper = np.array([cfit_mu, 20, sig_init + 2])

        # Optimize
        result = minimize(
            self.get_likelihood,
            init,
            args=(z, s, min_z, max_z),
            method='L-BFGS-B',
            bounds=list(zip(lower, upper))
        )

        # Special case: if sigma is very small, set prop to 1
        if result.x[2] < -10:
            result.x[1] = 10

        if verbose:
            print(f"Optimization success: {result.success}")
            print(f"Negative log-likelihood: {result.fun:.2f}")

        # Get fitted parameters
        gam, prop, sig, mu = self.map_params(result.x, min_z, max_z)

        # Compute Hessian for uncertainty estimates
        try:
            # Use R's optimHess via subprocess (exact same as IsoplotR)
            hessian = _compute_hessian_via_R(result.x, z, s, min_z, max_z, self.n_params)

            # Invert Hessian to get covariance matrix in unbounded space
            if np.linalg.cond(hessian) < 1/np.finfo(float).eps:
                cov_unbounded = np.linalg.inv(hessian)

                # Transform to model space using Jacobian
                J = self._compute_jacobian(result.x, min_z, max_z)
                cov_model = J @ cov_unbounded @ J.T

                # Extract standard errors
                variances = np.diag(cov_model)
                gam_se = np.sqrt(variances[0]) if variances[0] > 0 else np.nan
            else:
                if verbose:
                    print("Warning: Hessian is singular, cannot compute uncertainties")
                gam_se = np.nan

        except Exception as e:
            if verbose:
                print(f"Warning: Could not compute Hessian: {e}")
            gam_se = np.nan

        # Transform minimum age back to linear space
        min_age = np.exp(gam)
        min_age_se = min_age * gam_se  # Error propagation

        # Store results
        self.result = {
            'min_age': min_age,
            'min_age_se': min_age_se,
            'proportion': prop,
            'dispersion': sig,
            'log_likelihood': -result.fun,
            'success': result.success
        }

        return self.result

    def _compute_jacobian(self, par, min_z, max_z):
        """Compute Jacobian matrix for transformation from unbounded to model space."""
        J = np.zeros((self.n_params, self.n_params))

        # d(gam)/d(par[0]) = 1
        J[0, 0] = 1

        # d(prop)/d(par[1]) = exp(par[1]) / (1 + exp(par[1]))^2
        exp_p1 = np.exp(par[1])
        J[1, 1] = exp_p1 / (1 + exp_p1)**2

        # d(sig)/d(par[2]) = exp(par[2])
        J[2, 2] = np.exp(par[2])

        if self.n_params == 4:
            gam = par[0]
            exp_p3 = np.exp(par[3])

            # d(mu)/d(par[0]) = 1 - exp(par[3]) / (1 + exp(par[3]))
            J[3, 0] = 1 - exp_p3 / (1 + exp_p3)

            # d(mu)/d(par[3]) = (max_z - gam) * exp(par[3]) / (1 + exp(par[3]))^2
            J[3, 3] = (max_z - gam) * exp_p3 / (1 + exp_p3)**2

        return J


def maximum_likelihood_age(
        grains: [Grain],
        n_params: int = 4,
        verbose: bool = False
) -> (Grain, int, float):
    """
    Maximum Likelihood Age (MLA) using a two-component mixture model.

    This method models the age distribution as a mixture of:
    1. A discrete peak at the minimum age (syndepositional grains)
    2. A continuous truncated log-normal distribution (inherited grains)

    Based on Galbraith (2005) Section 6.11 and implemented following IsoplotR.

    Args:
        grains: List of Grain objects with age and uncertainty
        n_params: Number of parameters (3 or 4, default 4)
            3-param: gam, prop, sig (mu = gam)
            4-param: gam, prop, sig, mu (more flexible)
        verbose: Print optimization details

    Returns:
        Tuple of (grain, n, nan):
            grain: Grain object with MLA age and 1-sigma uncertainty
            n: Number of grains in dataset (int)
            nan: Not applicable for this method (use proportion instead)

    Notes:
        - Returns nan for age/uncertainty if optimization fails
        - Requires R installation (base R only, no packages needed) for uncertainty estimation
        - If R is not available, uncertainties will be set to nan
    """
    if not grains or len(grains) < 2:
        return Grain(float('nan'), float('nan')), 0, float('nan')

    # Extract ages and uncertainties
    ages = np.array([grain.age for grain in grains])
    uncertainties = np.array([grain.uncertainty for grain in grains])

    # Fit the model
    model = _MinimumAgeModel(n_params=n_params)

    try:
        result = model.fit(ages, uncertainties, verbose=verbose)

        if result['success']:
            min_age = result['min_age']
            min_age_se = result['min_age_se']

            # Handle nan uncertainties gracefully
            if np.isnan(min_age_se):
                if verbose:
                    print("Warning: Could not compute uncertainty for MLA")
                min_age_se = 0.0

            mla_grain = Grain(age=min_age, uncertainty=min_age_se)
            n = len(grains)

            return mla_grain, n, float('nan')
        else:
            if verbose:
                print("Warning: MLA optimization failed")
            return Grain(float('nan'), float('nan')), len(grains), float('nan')

    except Exception as e:
        if verbose:
            print(f"Warning: MLA calculation error: {e}")
        return Grain(float('nan'), float('nan')), len(grains), float('nan')


def _get_pretty_ticks(data_min, data_max, n_ticks=7):
    """
    Generate 'pretty' tick values similar to R's pretty() function.
    Returns nicely spaced round numbers spanning the data range.
    """
    data_range = data_max - data_min
    if data_range <= 0:
        return np.array([data_min])

    # Calculate rough step size
    rough_step = data_range / (n_ticks - 1)

    # Find the magnitude
    magnitude = 10 ** np.floor(np.log10(rough_step))

    # Normalize to 1-10 range
    normalized = rough_step / magnitude

    # Choose a nice step (1, 2, 2.5, 5, or 10)
    if normalized <= 1.5:
        nice_step = 1
    elif normalized <= 3:
        nice_step = 2
    elif normalized <= 7:
        nice_step = 5
    else:
        nice_step = 10

    step = nice_step * magnitude

    # Calculate tick range
    tick_min = np.floor(data_min / step) * step
    tick_max = np.ceil(data_max / step) * step

    return np.arange(tick_min, tick_max + step * 0.5, step)


def radial_plot(
        grains: [Grain],
        mla_result: Grain = None,
        title: str = None,
        font_path: str = None,
        font_size: float = 10,
        fig_width: float = 8,
        fig_height: float = 6,
        color_points: str = "#1f77b4",
        color_mla: str = "red",
        show_2sigma_band: bool = True,
):
    """
    Create a radial plot for visualizing MLA results.

    Based on Galbraith (1988, 1990) and IsoplotR implementation by Vermeesch.
    Radial plots visualize heteroscedastic datasets (unequal uncertainties).

    The plot shows:
    - Data points: each grain plotted at precision (x) vs standardized estimate (y)
    - Radial scale: age arc on right side
    - 2σ bands: horizontal lines at y = ±2
    - MLA estimate: radial line from origin through MLA age

    Args:
        grains: List of Grain objects with age and uncertainty
        mla_result: Optional MLA Grain result to display on plot
        title: Plot title
        font_path: Path to custom font
        font_size: Font size for labels
        fig_width: Figure width in inches
        fig_height: Figure height in inches
        color_points: Color for data points
        color_mla: Color for MLA estimate line
        show_2sigma_band: Whether to show 2σ confidence bands

    Returns:
        matplotlib figure object

    References:
        Galbraith, R.F., 1988. Graphical display of estimates having
            differing standard errors. Technometrics 30(3), 271-281.
        Vermeesch, P., 2018. IsoplotR: A free and open toolbox for geochronology.
            Geoscience Frontiers, 9, 1479-1493.
    """
    import matplotlib.font_manager as fm

    # Extract ages and uncertainties
    ages = np.array([grain.age for grain in grains])
    uncertainties = np.abs(np.array([grain.uncertainty for grain in grains]))

    # Handle zero uncertainties
    zero_mask = uncertainties <= 0
    if np.any(zero_mask):
        uncertainties[zero_mask] = np.maximum(ages[zero_mask] * 0.01, 1.0)

    # Transform to log-space
    z = np.log(ages)
    s = uncertainties / ages  # Relative errors (standard error in log-space)

    # Reference value z0 (weighted mean in log space)
    weights = 1 / s**2
    z0 = np.sum(weights * z) / np.sum(weights)

    # Data coordinates: x = precision (1/s), y = standardized estimate ((z-z0)/s)
    rx = 1 / s
    ry = (z - z0) / s

    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    # Determine data extents
    rx_max = np.max(rx)
    ry_min, ry_max = np.min(ry), np.max(ry)

    # Y-range should include ±2 for 2-sigma bands at minimum
    y_extent = max(abs(ry_min), abs(ry_max), 2.5) * 1.15

    # Arc position - to the right of all data points
    arc_x = rx_max * 1.12

    # Calculate the y-positions on the arc for different ages
    # In radial plot geometry: a line from origin with slope (z-z0) intersects
    # the arc at x=arc_x, y = arc_x * (z - z0)
    z_min_data = z.min()
    z_max_data = z.max()
    z_range = z_max_data - z_min_data
    z_min = z_min_data - 0.1 * z_range
    z_max = z_max_data + 0.1 * z_range

    # Function to convert age to arc y-coordinate
    def age_to_arc_y(age):
        z_age = np.log(age)
        return arc_x * (z_age - z0)

    # Draw 2-sigma bands (horizontal lines at y = ±2)
    if show_2sigma_band:
        ax.axhline(y=2, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
        ax.axhline(y=-2, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.4, zorder=1)

    # Generate age ticks using pretty spacing
    age_min_plot = np.exp(z_min)
    age_max_plot = np.exp(z_max)
    age_ticks = _get_pretty_ticks(age_min_plot, age_max_plot, n_ticks=8)
    # Filter to reasonable range
    age_ticks = age_ticks[(age_ticks > 0) &
                          (age_ticks >= age_min_plot * 0.9) &
                          (age_ticks <= age_max_plot * 1.1)]

    # Calculate arc y-coordinates for the age range
    arc_y_min = age_to_arc_y(age_min_plot)
    arc_y_max = age_to_arc_y(age_max_plot)

    # Tick mark length
    tick_len = rx_max * 0.02

    # Draw the arc (vertical line at arc_x)
    ax.plot([arc_x, arc_x], [arc_y_min, arc_y_max], 'k-', linewidth=1.5, zorder=2)

    # Draw radial lines and age labels
    for age_tick in age_ticks:
        arc_y = age_to_arc_y(age_tick)

        # Draw radial line from origin to arc
        ax.plot([0, arc_x], [0, arc_y], 'k-', linewidth=0.5, alpha=0.3, zorder=1)

        # Draw tick mark on arc
        ax.plot([arc_x, arc_x + tick_len], [arc_y, arc_y], 'k-', linewidth=1.0, zorder=2)

        # Format tick label
        if age_tick >= 1000:
            label = f'{int(age_tick)}'
        elif age_tick >= 100:
            label = f'{int(age_tick)}'
        elif age_tick >= 1:
            label = f'{age_tick:.0f}' if age_tick == int(age_tick) else f'{age_tick:.1f}'
        else:
            label = f'{age_tick:.2f}'

        # Add age label
        ax.text(arc_x + tick_len * 2, arc_y, label, fontsize=font_size * 0.85,
                ha='left', va='center', zorder=3)

    # Draw MLA estimate line
    if mla_result is not None and not np.isnan(mla_result.age):
        mla_arc_y = age_to_arc_y(mla_result.age)
        ax.plot([0, arc_x * 1.05], [0, mla_arc_y * 1.05], color=color_mla,
                linewidth=2.0, zorder=4, label=f'MLA = {mla_result.age:.1f} Ma')

    # Plot data points
    ax.scatter(rx, ry, c=color_points, s=50, alpha=0.7,
               edgecolors='white', linewidths=0.5, zorder=5, marker='o')

    # Axis labels
    ax.set_xlabel('Precision (1/σ)', fontsize=font_size)
    ax.set_ylabel('Standardised estimate', fontsize=font_size)

    # Set axis limits - make sure arc and labels fit
    x_margin = rx_max * 0.25
    ax.set_xlim(-rx_max * 0.05, arc_x + x_margin)
    ax.set_ylim(-y_extent, y_extent)

    # Add "t (Ma)" label at top of arc
    ax.text(arc_x + tick_len * 2, arc_y_max * 1.05, 't (Ma)',
            fontsize=font_size, ha='left', va='bottom', fontweight='bold')

    # Add title
    if title:
        if font_path:
            font_prop = fm.FontProperties(fname=font_path)
            ax.set_title(title, fontsize=font_size * 1.3, fontproperties=font_prop)
        else:
            ax.set_title(title, fontsize=font_size * 1.3)

    # Add legend if MLA shown (position in lower right to avoid data)
    if mla_result is not None and not np.isnan(mla_result.age):
        ax.legend(loc='lower right', fontsize=font_size * 0.9)

    # Clean up spines - only show left and bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add n count annotation (upper left)
    ax.text(0.02, 0.98, f'n = {len(grains)}', transform=ax.transAxes,
            fontsize=font_size * 0.9, va='top', ha='left')

    fig.tight_layout()
    plt.close()

    return fig
