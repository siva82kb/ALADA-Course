"""
Script to demonstrate constrained least squares line fitting.

- Left panel  : scatter plot of N random (x, y) points with the constrained
                best-fit line, a hover-preview line, and an optionally
                selected line.
- Right panel : contour of J(m, b); the constraint c1·m + c2·b = d is drawn
                as a line; the constrained optimum is marked; hovering shows a
                preview marker that slides along the constraint line.
- Hover       : mouse x-position (or y-position when c2=0) projects onto the
                constraint line and previews the corresponding straight line.
- Click       : locks the hovered point as the selected solution.

Controls
--------
  "Generate Points" button : draw 20 new random points and reset.
  c1 / c2 / d text boxes   : set the constraint coefficients (press Enter).
  Hover on contour          : preview solutions on the constraint line.
  Click on contour          : select a solution.
  Escape                    : close the window.

Author: Sivakumar Balasubramanian
Date: 07 April 2026
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
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
N        = 20    # number of data points
GRID_RES = 120   # contour grid resolution (lower keeps hover snappy)

OPT_COLOR   = 'tab:green'   # constrained optimum
HOV_COLOR   = '#888888'     # hover preview
SEL_COLOR   = 'tab:red'     # selected (clicked) point
CON_COLOR   = '#222222'     # constraint line on contour

CONTOUR_M_LIM = (-6.0, 6.0)
CONTOUR_B_LIM = (-6.0, 6.0)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
x_data = None
y_data = None

c1_val = 1.0
c2_val = 1.0
d_val  = 0.0

m_opt = None    # constrained optimum slope
b_opt = None    # constrained optimum intercept
J_opt = None

hover_m = None
hover_b = None

selected_m = None
selected_b = None
J_sel      = None

scatter_xlim = None
scatter_ylim = None

# Cached contour grid (recomputed only when data changes)
_M_grid = None
_B_grid = None
_J_grid = None


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def objective(m, b):
    return float(np.sum((y_data - (m * x_data + b)) ** 2))


def _build_grid():
    global _M_grid, _B_grid, _J_grid
    m_vals = np.linspace(CONTOUR_M_LIM[0], CONTOUR_M_LIM[1], GRID_RES)
    b_vals = np.linspace(CONTOUR_B_LIM[0], CONTOUR_B_LIM[1], GRID_RES)
    M, B = np.meshgrid(m_vals, b_vals)
    pred = (M[:, :, np.newaxis] * x_data[np.newaxis, np.newaxis, :]
            + B[:, :, np.newaxis])
    _J_grid = np.sum((y_data[np.newaxis, np.newaxis, :] - pred) ** 2, axis=2)
    _M_grid, _B_grid = M, B


def _project_to_constraint(mx, by):
    """
    Project a mouse position onto the constraint line c1·m + c2·b = d.
    Uses mx (x-position) when c2 != 0, by (y-position) when c2 == 0.
    Returns (m, b) or (None, None) for a degenerate constraint.
    """
    if abs(c2_val) > 1e-10:
        m = mx
        b = (d_val - c1_val * m) / c2_val
    elif abs(c1_val) > 1e-10:
        b = by
        m = d_val / c1_val
    else:
        return None, None
    return m, b


def _in_contour_bounds(m, b):
    return (CONTOUR_M_LIM[0] <= m <= CONTOUR_M_LIM[1] and
            CONTOUR_B_LIM[0] <= b <= CONTOUR_B_LIM[1])


def compute_constrained_opt():
    """Solve min ||Aθ - y||² s.t. Cθ = d via KKT block system."""
    global m_opt, b_opt, J_opt
    if abs(c1_val) < 1e-10 and abs(c2_val) < 1e-10:
        m_opt = b_opt = J_opt = None
        return
    if x_data is None:
        m_opt = b_opt = J_opt = None
        return
    A   = np.column_stack([x_data, np.ones(N)])
    AtA = A.T @ A
    Aty = A.T @ y_data
    C   = np.array([[c1_val, c2_val]])
    KKT = np.block([[AtA, C.T], [C, np.zeros((1, 1))]])
    rhs = np.append(Aty, d_val)
    try:
        sol   = np.linalg.solve(KKT, rhs)
        m_opt = float(sol[0])
        b_opt = float(sol[1])
        J_opt = objective(m_opt, b_opt)
    except np.linalg.LinAlgError:
        m_opt = b_opt = J_opt = None


# ---------------------------------------------------------------------------
# Event callbacks
# ---------------------------------------------------------------------------
def generate_new_points(event=None):
    global x_data, y_data, selected_m, selected_b, J_sel
    global scatter_xlim, scatter_ylim, hover_m, hover_b

    m_true = np.random.uniform(-2, 2)
    b_true = np.random.uniform(-2, 2)
    x_data = np.random.uniform(-3, 3, N)
    y_data = m_true * x_data + b_true + np.random.randn(N) * 0.8

    compute_constrained_opt()
    _build_grid()

    # Fix scatter limits from data + optimal line (if valid)
    xpad = 0.5
    xmin = float(x_data.min()) - xpad
    xmax = float(x_data.max()) + xpad
    xline = np.linspace(xmin, xmax, 300)
    if m_opt is not None:
        all_y = np.concatenate([y_data, m_opt * xline + b_opt])
    else:
        all_y = y_data.copy()
    ypad = 0.5
    scatter_xlim = (xmin, xmax)
    scatter_ylim = (float(all_y.min()) - ypad, float(all_y.max()) + ypad)

    selected_m = selected_b = J_sel = None
    hover_m = hover_b = None
    redraw()


def _update_constraint(new_c1=None, new_c2=None, new_d=None):
    """Called by any TextBox submit — updates constraint and redraws."""
    global c1_val, c2_val, d_val, selected_m, selected_b, J_sel, hover_m, hover_b
    if new_c1 is not None:
        c1_val = new_c1
    if new_c2 is not None:
        c2_val = new_c2
    if new_d is not None:
        d_val = new_d
    selected_m = selected_b = J_sel = None
    hover_m = hover_b = None
    compute_constrained_opt()
    redraw()


def on_c1_submit(text):
    try:
        _update_constraint(new_c1=float(text))
    except ValueError:
        pass


def on_c2_submit(text):
    try:
        _update_constraint(new_c2=float(text))
    except ValueError:
        pass


def on_d_submit(text):
    try:
        _update_constraint(new_d=float(text))
    except ValueError:
        pass


def on_hover(event):
    global hover_m, hover_b
    if event.inaxes is not ax_contour or x_data is None:
        if hover_m is not None:
            hover_m = hover_b = None
            redraw()
        return
    m, b = _project_to_constraint(event.xdata, event.ydata)
    if m is None or not _in_contour_bounds(m, b):
        if hover_m is not None:
            hover_m = hover_b = None
            redraw()
        return
    hover_m, hover_b = m, b
    redraw()


def on_click(event):
    global selected_m, selected_b, J_sel
    if event.inaxes is not ax_contour or x_data is None:
        return
    m, b = _project_to_constraint(event.xdata, event.ydata)
    if m is None or not _in_contour_bounds(m, b):
        return
    selected_m, selected_b = m, b
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
    assert x_data is not None and scatter_xlim is not None
    ax_scatter.cla()
    xline = np.linspace(scatter_xlim[0], scatter_xlim[1], 300)

    # Data points
    ax_scatter.scatter(x_data, y_data, color='#333333', s=35, zorder=4,
                       label='Data')

    # Hover preview line (drawn under selected)
    if hover_m is not None:
        ax_scatter.plot(xline, hover_m * xline + hover_b,
                        color=HOV_COLOR, lw=1.5, ls='--', zorder=2,
                        alpha=0.7, label='Preview')

    # Selected line
    if selected_m is not None:
        ax_scatter.plot(xline, selected_m * xline + selected_b,
                        color=SEL_COLOR, lw=2.0, ls='--', zorder=3,
                        label='Selected')

    # Constrained optimal line (on top)
    if m_opt is not None:
        ax_scatter.plot(xline, m_opt * xline + b_opt,
                        color=OPT_COLOR, lw=2.5, zorder=5,
                        label='Constrained opt.')

    ax_scatter.set_xlim(*scatter_xlim)
    ax_scatter.set_ylim(*scatter_ylim)
    ax_scatter.axhline(0, color='#cccccc', lw=0.8, ls=':', zorder=1)
    ax_scatter.axvline(0, color='#cccccc', lw=0.8, ls=':', zorder=1)
    ax_scatter.spines['right'].set_visible(False)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.set_xlabel('$x$', fontsize=13)
    ax_scatter.set_ylabel('$y$', fontsize=13)
    ax_scatter.set_title('Data & Fitted Lines', fontsize=13, pad=8)
    ax_scatter.legend(fontsize=10, loc='best')


def _draw_constraint_line():
    """Draw the constraint line c1·m + c2·b = d on the contour axes."""
    if abs(c1_val) < 1e-10 and abs(c2_val) < 1e-10:
        return
    # Extend well beyond limits so matplotlib clips to the axes bounds
    big = 100.0
    if abs(c2_val) > 1e-10:
        m_pts = np.array([-big, big])
        b_pts = (d_val - c1_val * m_pts) / c2_val
    else:
        m_fixed = d_val / c1_val
        m_pts = np.array([m_fixed, m_fixed])
        b_pts = np.array([-big, big])
    ax_contour.plot(m_pts, b_pts, color=CON_COLOR, lw=1.5, ls='-',
                    zorder=3, label='Constraint')


def draw_contour():
    ax_contour.cla()

    # Contour lines from cached grid
    if _J_grid is not None:
        ax_contour.contour(_M_grid, _B_grid, _J_grid,
                           levels=20, cmap='RdYlBu_r', linewidths=1.2)

    # Ellipse tangent to constraint at the hover point (dotted grey)
    if _J_grid is not None and hover_m is not None:
        J_hov = objective(hover_m, hover_b)
        ax_contour.contour(_M_grid, _B_grid, _J_grid,
                           levels=[J_hov], colors=[HOV_COLOR],
                           linewidths=1.2, linestyles=':', alpha=0.7, zorder=5)

    # Ellipse for the selected point (dotted thin red)
    if _J_grid is not None and selected_m is not None:
        J_s = objective(selected_m, selected_b)
        ax_contour.contour(_M_grid, _B_grid, _J_grid,
                           levels=[J_s], colors=[SEL_COLOR],
                           linewidths=1.2, linestyles=':', alpha=0.7, zorder=5)

    # Ellipse tangent to constraint at the constrained optimum (solid red)
    if _J_grid is not None and J_opt is not None:
        ax_contour.contour(_M_grid, _B_grid, _J_grid,
                           levels=[J_opt], colors=['tab:red'],
                           linewidths=2.0, zorder=6)

    # Constraint line
    _draw_constraint_line()

    # Constrained optimum marker + drop-lines
    if m_opt is not None:
        ax_contour.plot(m_opt, b_opt,
                        marker='*', color=OPT_COLOR, markersize=12, zorder=8,
                        markeredgecolor='#444444', markeredgewidth=0.5,
                        label='Constrained opt.', linestyle='None')
        ax_contour.plot([m_opt, m_opt], [CONTOUR_B_LIM[0], b_opt],
                        color=OPT_COLOR, lw=1.0, ls=':', zorder=7)
        ax_contour.plot([CONTOUR_M_LIM[0], m_opt], [b_opt, b_opt],
                        color=OPT_COLOR, lw=1.0, ls=':', zorder=7)

    # Hover preview marker + lighter drop-lines
    if hover_m is not None:
        ax_contour.plot(hover_m, hover_b,
                        marker='o', color=HOV_COLOR, markersize=7, zorder=8,
                        markeredgecolor='#555555', markeredgewidth=0.5,
                        alpha=0.8, label='Preview', linestyle='None')
        ax_contour.plot([hover_m, hover_m], [CONTOUR_B_LIM[0], hover_b],
                        color=HOV_COLOR, lw=0.7, ls=':', alpha=0.4, zorder=7)
        ax_contour.plot([CONTOUR_M_LIM[0], hover_m], [hover_b, hover_b],
                        color=HOV_COLOR, lw=0.7, ls=':', alpha=0.4, zorder=7)

    # Selected marker + drop-lines
    if selected_m is not None:
        ax_contour.plot(selected_m, selected_b,
                        marker='o', color=SEL_COLOR, markersize=9, zorder=8,
                        markeredgecolor='#333333', markeredgewidth=0.5,
                        label='Selected', linestyle='None')
        ax_contour.plot([selected_m, selected_m], [CONTOUR_B_LIM[0], selected_b],
                        color=SEL_COLOR, lw=0.7, ls=':', alpha=0.5, zorder=7)
        ax_contour.plot([CONTOUR_M_LIM[0], selected_m], [selected_b, selected_b],
                        color=SEL_COLOR, lw=0.7, ls=':', alpha=0.5, zorder=7)

    ax_contour.set_xlim(*CONTOUR_M_LIM)
    ax_contour.set_ylim(*CONTOUR_B_LIM)
    ax_contour.set_aspect('equal', adjustable='box')
    ax_contour.set_xlabel('$m$  (slope)', fontsize=13)
    ax_contour.set_ylabel('$b$  (intercept)', fontsize=13)
    ax_contour.set_title('Objective $J(m,\\,b)$', fontsize=13, pad=8)
    ax_contour.legend(fontsize=9, loc='upper right')


def draw_text():
    ax_text.cla()
    ax_text.axis('off')
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    if x_data is None:
        return

    # Constraint equation
    c_str = (f"Constraint:  "
             f"${c1_val:g}\\cdot m + {c2_val:g}\\cdot b = {d_val:g}$")
    ax_text.text(0.01, 0.65, c_str, fontsize=11, color='#222222',
                 va='center', transform=ax_text.transAxes)

    # Constrained optimum
    if m_opt is not None:
        opt_str = (f"Constrained opt.:  "
                   f"$m^* = {m_opt:+.3f}$,   "
                   f"$b^* = {b_opt:+.3f}$,   "
                   f"$J^* = {J_opt:.3f}$")
        ax_text.text(0.01, 0.50, opt_str, fontsize=11,
                     color=OPT_COLOR, fontweight='bold',
                     va='center', transform=ax_text.transAxes)
    else:
        ax_text.text(0.01, 0.50, 'Degenerate constraint (c₁ = c₂ = 0).',
                     fontsize=11, color='tab:red', va='center',
                     transform=ax_text.transAxes)

    # Selected / hover / hint
    if selected_m is not None:
        J_s = objective(selected_m, selected_b)
        sel_str = (f"Selected:  "
                   f"$m = {selected_m:+.3f}$,   "
                   f"$b = {selected_b:+.3f}$,   "
                   f"$J = {J_s:.3f}$")
        ax_text.text(0.01, 0.18, sel_str, fontsize=11, color=SEL_COLOR,
                     va='center', transform=ax_text.transAxes)
    elif hover_m is not None:
        J_h = objective(hover_m, hover_b)
        hov_str = (f"Preview:  "
                   f"$m = {hover_m:+.3f}$,   "
                   f"$b = {hover_b:+.3f}$,   "
                   f"$J = {J_h:.3f}$")
        ax_text.text(0.01, 0.18, hov_str, fontsize=11, color=HOV_COLOR,
                     va='center', transform=ax_text.transAxes)
    else:
        ax_text.text(0.01, 0.18,
                     'Hover over the contour plot to preview a solution along the constraint.',
                     fontsize=10, color='#999999', style='italic',
                     va='center', transform=ax_text.transAxes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig = plt.figure(figsize=(13, 8.0))
    fig.canvas.manager.set_window_title(
        'ALADA – Constrained Least Squares Line Fitting')

    # Main axes
    ax_scatter = fig.add_axes([0.06, 0.24, 0.38, 0.70])
    ax_contour = fig.add_axes([0.55, 0.24, 0.38, 0.70])
    ax_text    = fig.add_axes([0.06, 0.01, 0.90, 0.12])
    ax_text.axis('off')

    # -----------------------------------------------------------------------
    # Generate button — below the scatter plot (left side)
    # -----------------------------------------------------------------------
    ax_btn = fig.add_axes([0.10, 0.118, 0.10, 0.062])
    btn = Button(ax_btn, 'Generate\nPoints', color='#d0e8f5', hovercolor='#a0c8ef')
    btn.on_clicked(generate_new_points)

    # -----------------------------------------------------------------------
    # Constraint controls — below the contour plot (right side)
    # x range of contour: 0.55 → 0.93
    # -----------------------------------------------------------------------
    fig.text(0.555, 0.149, 'Constraint:', fontsize=11,
             va='center', ha='left', color='#222222')

    ax_c1 = fig.add_axes([0.648, 0.118, 0.055, 0.052])
    tb_c1 = TextBox(ax_c1, '', initial='1.0', textalignment='center')
    tb_c1.on_submit(on_c1_submit)
    fig.text(0.675, 0.178, '$c_1$', fontsize=12,
             va='bottom', ha='center', color='#555555')

    fig.text(0.708, 0.149, '$\\cdot\\,m\\;+$', fontsize=11,
             va='center', ha='left', color='#222222')

    ax_c2 = fig.add_axes([0.762, 0.118, 0.055, 0.052])
    tb_c2 = TextBox(ax_c2, '', initial='1.0', textalignment='center')
    tb_c2.on_submit(on_c2_submit)
    fig.text(0.789, 0.178, '$c_2$', fontsize=12,
             va='bottom', ha='center', color='#555555')

    fig.text(0.822, 0.149, '$\\cdot\\,b\\;=$', fontsize=11,
             va='center', ha='left', color='#222222')

    ax_d = fig.add_axes([0.873, 0.118, 0.055, 0.052])
    tb_d = TextBox(ax_d, '', initial='0.0', textalignment='center')
    tb_d.on_submit(on_d_submit)
    fig.text(0.900, 0.178, '$d$', fontsize=12,
             va='bottom', ha='center', color='#555555')

    # -----------------------------------------------------------------------
    # Event connections
    # -----------------------------------------------------------------------
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial draw
    generate_new_points()

    plt.show()
