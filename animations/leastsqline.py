"""
Script to demonstrate unconstrained least squares line fitting.

- Left panel  : scatter plot of N random (x, y) points with the best-fit line
                and an optionally selected line.
- Right panel : filled contour of the least-squares objective J(m, b) as a
                function of slope m and intercept b; the optimal point is
                marked with a gold star, and any point chosen by clicking
                is marked with a red circle.
- Clicking the contour selects a (m, b) pair whose line is overlaid on the
  scatter plot.
- The objective value at the optimum (J*) and at the selected point (J_sel)
  are displayed at the bottom.

Controls
--------
  "Generate Points" button : draw 20 new random points and reset.
  Click on right panel     : select (m, b) and update the line.
  Escape                   : close the window.

Author: Sivakumar Balasubramanian
Date: 07 April 2026
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import platform

# ---------------------------------------------------------------------------
# Font / toolbar setup
# ---------------------------------------------------------------------------
if platform.system() == "Windows":
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
else:
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans', 'Liberation Sans']
mpl.rcParams['toolbar'] = 'None'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N = 20                  # number of data points
GRID_RES = 200          # contour grid resolution

OPT_COLOR = 'tab:orange'   # colour for the optimal point and its line
SEL_COLOR = 'tab:red'      # colour for the selected point and its line

# Fixed axis ranges — constant across all datasets
CONTOUR_M_LIM = (-6.0, 6.0)   # slope axis range for contour plot
CONTOUR_B_LIM = (-6.0, 6.0)   # intercept axis range for contour plot

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
x_data = None
y_data = None
m_opt  = None
b_opt  = None
J_opt  = None
selected_m = None
selected_b = None
J_sel      = None
scatter_xlim = None    # fixed for the current dataset
scatter_ylim = None    # fixed for the current dataset


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def objective(m, b):
    """Scalar least-squares cost for a given (m, b)."""
    return float(np.sum((y_data - (m * x_data + b)) ** 2))


def objective_grid(m_vals, b_vals):
    """Vectorised J over a 2-D grid; returns (M, B, J) arrays."""
    M, B = np.meshgrid(m_vals, b_vals)
    # pred shape: (n_b, n_m, N)
    pred = (M[:, :, np.newaxis] * x_data[np.newaxis, np.newaxis, :]
            + B[:, :, np.newaxis])
    J = np.sum((y_data[np.newaxis, np.newaxis, :] - pred) ** 2, axis=2)
    return M, B, J


# ---------------------------------------------------------------------------
# Event callbacks
# ---------------------------------------------------------------------------
def generate_new_points(event=None):
    """Generate N random points along a noisy line and compute the optimum."""
    global x_data, y_data, m_opt, b_opt, J_opt
    global selected_m, selected_b, J_sel
    global scatter_xlim, scatter_ylim

    m_true = np.random.uniform(-2, 2)
    b_true = np.random.uniform(-2, 2)
    x_data = np.random.uniform(-3, 3, N)
    y_data = m_true * x_data + b_true + np.random.randn(N) * 0.8

    A = np.column_stack([x_data, np.ones(N)])
    theta = np.linalg.lstsq(A, y_data, rcond=None)[0]
    m_opt, b_opt = float(theta[0]), float(theta[1])
    J_opt = objective(m_opt, b_opt)

    # Compute scatter limits once for this dataset (based on data + optimal line)
    xpad = 0.5
    xmin, xmax = float(x_data.min()) - xpad, float(x_data.max()) + xpad
    xline = np.linspace(xmin, xmax, 300)
    opt_y = m_opt * xline + b_opt
    all_y = np.concatenate([y_data, opt_y])
    ypad = 0.5
    scatter_xlim = (xmin, xmax)
    scatter_ylim = (float(all_y.min()) - ypad, float(all_y.max()) + ypad)

    # Reset any previous selection
    selected_m = selected_b = J_sel = None

    redraw()


def on_contour_click(event):
    """Pick a (m, b) by clicking on the contour plot."""
    global selected_m, selected_b, J_sel
    if event.inaxes is not ax_contour or x_data is None:
        return
    selected_m = event.xdata
    selected_b = event.ydata
    J_sel = objective(selected_m, selected_b)
    redraw()


def on_key(event):
    if event.key == 'escape':
        plt.close(fig)


# ---------------------------------------------------------------------------
# Drawing routines
# ---------------------------------------------------------------------------
def redraw():
    if x_data is None:
        return
    draw_scatter()
    draw_contour()
    draw_text()
    fig.canvas.draw_idle()


def draw_scatter():
    assert x_data is not None and y_data is not None
    assert m_opt is not None and b_opt is not None
    assert scatter_xlim is not None and scatter_ylim is not None
    ax_scatter.cla()

    xline = np.linspace(scatter_xlim[0], scatter_xlim[1], 300)

    # Data points
    ax_scatter.scatter(x_data, y_data, color='#333333', s=35, zorder=3,
                       label='Data')

    # Optimal line (colour matches the star on the contour)
    ax_scatter.plot(xline, m_opt * xline + b_opt,
                    color=OPT_COLOR, lw=2.5, zorder=2, label='Best fit')

    # Selected line (shown only after a click)
    if selected_m is not None:
        ax_scatter.plot(xline, selected_m * xline + selected_b,
                        color=SEL_COLOR, lw=2, zorder=2, linestyle='--',
                        label='Selected')

    # Fixed limits for this dataset
    ax_scatter.set_xlim(*scatter_xlim)
    ax_scatter.set_ylim(*scatter_ylim)

    # Style
    ax_scatter.spines['right'].set_visible(False)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.set_xlabel('$x$', fontsize=13)
    ax_scatter.set_ylabel('$y$', fontsize=13)
    ax_scatter.set_title('Data & Fitted Lines', fontsize=13, pad=8)
    ax_scatter.legend(fontsize=10, loc='best')


def draw_contour():
    assert m_opt is not None and b_opt is not None
    ax_contour.cla()

    # Fixed grid over the constant axis ranges
    m_vals = np.linspace(CONTOUR_M_LIM[0], CONTOUR_M_LIM[1], GRID_RES)
    b_vals = np.linspace(CONTOUR_B_LIM[0], CONTOUR_B_LIM[1], GRID_RES)
    M, B, J = objective_grid(m_vals, b_vals)

    # Contour lines coloured by objective value (low = blue, high = red)
    ax_contour.contour(M, B, J, levels=20, cmap='RdYlBu_r', linewidths=1.2)

    # Optimal marker (same colour as the line in the scatter plot)
    ax_contour.plot(m_opt, b_opt,
                    marker='*', color=OPT_COLOR, markersize=12, zorder=5,
                    markeredgecolor='#444444', markeredgewidth=0.5,
                    label='Optimal', linestyle='None')
    # Drop lines from optimal point to axes
    ax_contour.plot([m_opt, m_opt], [CONTOUR_B_LIM[0], b_opt],
                    color=OPT_COLOR, lw=1.0, ls=':', zorder=4)
    ax_contour.plot([CONTOUR_M_LIM[0], m_opt], [b_opt, b_opt],
                    color=OPT_COLOR, lw=1.0, ls=':', zorder=4)

    # Selected marker
    if selected_m is not None:
        ax_contour.plot(selected_m, selected_b,
                        marker='o', color=SEL_COLOR, markersize=8, zorder=5,
                        markeredgecolor='#333333', markeredgewidth=0.5,
                        label='Selected', linestyle='None')
        # Drop lines from selected point to axes (lighter)
        ax_contour.plot([selected_m, selected_m], [CONTOUR_B_LIM[0], selected_b],
                        color=SEL_COLOR, lw=0.7, ls=':', alpha=0.5, zorder=4)
        ax_contour.plot([CONTOUR_M_LIM[0], selected_m], [selected_b, selected_b],
                        color=SEL_COLOR, lw=0.7, ls=':', alpha=0.5, zorder=4)

    ax_contour.set_xlim(*CONTOUR_M_LIM)
    ax_contour.set_ylim(*CONTOUR_B_LIM)
    ax_contour.set_aspect('equal', adjustable='box')
    ax_contour.set_xlabel('$m$  (slope)', fontsize=13)
    ax_contour.set_ylabel('$b$  (intercept)', fontsize=13)
    ax_contour.set_title('Objective $J(m,\\,b)$', fontsize=13, pad=8)
    ax_contour.legend(fontsize=10, loc='upper right')


def draw_text():
    ax_text.cla()
    ax_text.axis('off')
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    if x_data is None or m_opt is None or b_opt is None or J_opt is None:
        return

    opt_str = (f"Optimal:   "
               f"$m^* = {m_opt:+.3f}$,  "
               f"$b^* = {b_opt:+.3f}$,  "
               f"$J^* = {J_opt:.3f}$")
    ax_text.text(0.02, 0.72, opt_str, fontsize=12,
                 color=OPT_COLOR, fontweight='bold', va='center',
                 transform=ax_text.transAxes)

    if selected_m is not None:
        sel_str = (f"Selected:  "
                   f"$m = {selected_m:+.3f}$,  "
                   f"$b = {selected_b:+.3f}$,  "
                   f"$J = {J_sel:.3f}$")
        ax_text.text(0.02, 0.22, sel_str, fontsize=12,
                     color=SEL_COLOR, va='center',
                     transform=ax_text.transAxes)
    else:
        ax_text.text(0.02, 0.22,
                     'Click on the contour plot to select (m, b).',
                     fontsize=11, color='#888888', style='italic', va='center',
                     transform=ax_text.transAxes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig = plt.figure(figsize=(13, 7.5))
    fig.canvas.manager.set_window_title('ALADA – Least Squares Line Fitting')

    # Axes layout (manual positioning for precise control)
    ax_scatter = fig.add_axes([0.06, 0.18, 0.40, 0.74])
    ax_contour = fig.add_axes([0.57, 0.18, 0.40, 0.74])
    ax_text    = fig.add_axes([0.06, 0.01, 0.88, 0.10])
    ax_text.axis('off')

    # Button centred between the two panels
    ax_btn = fig.add_axes([0.435, 0.04, 0.13, 0.08])
    btn = Button(ax_btn, 'Generate\nPoints', color='#d0e8f5', hovercolor='#a0c8ef')
    btn.on_clicked(generate_new_points)

    # Event connections
    fig.canvas.mpl_connect('button_press_event', on_contour_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Kick off with an initial set of points
    generate_new_points()

    plt.show()
