"""
Script to demonstrate constrained least squares fitting of a 2nd-order polynomial.

Polynomial model : y = a0 + a1·x + a2·x²
Constraint       : c1·a0 + c2·a1 + c3·a2 = d

The cost function J(a0, a1, a2) lives in 3D parameter space.  One linear
constraint reduces the feasible set to a 2D affine subspace (the constraint
plane).  The right panel visualises J *restricted to that plane*, parameterised as

    θ(s, t) = θ_p  +  s·v₁  +  t·v₂

where θ_p is the minimum-norm particular solution on the plane and {v₁, v₂}
form an orthonormal basis for the null space of C = [c1, c2, c3].  Every
point (s, t) in the contour plot therefore corresponds to a valid constrained
solution, and the constrained optimum is the bowl minimum on that plane.

- Left panel  : scatter plot of N random (x, y) points with the constrained
                best-fit polynomial (green), a hover-preview (grey), and an
                optionally selected curve (red).
- Right panel : 2D filled contour of J(s, t) on the constraint plane.
                The constrained optimum (s*, t*) is marked with a star.
- Hover       : mouse over the contour previews the polynomial θ(s, t).
- Click       : locks the hovered (s, t) as the selected solution.

Controls
--------
  "Generate Points" button   : draw 20 new random points and reset.
  c1 / c2 / c3 / d text boxes: update the constraint C·θ = d (press Enter).
  Hover on contour            : preview constrained solutions interactively.
  Click on contour            : select (lock) a solution.
  Escape                      : close the window.

Author: Sivakumar Balasubramanian
Date: 14 April 2026
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
N         = 20
GRID_RES  = 100   # resolution of the (s, t) contour grid
GRID_HALF = 4.0   # default half-range for the (s, t) axes

OPT_COLOR = 'tab:green'
HOV_COLOR = '#888888'
SEL_COLOR = 'tab:red'

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
x_data = None
y_data = None

c_val = np.array([1.0, 0.0, 0.0])   # constraint coefficient vector
d_val = 0.0                           # constraint RHS: C · θ = d

# Constraint-plane parameterisation (recomputed when constraint changes)
theta_p = None   # particular solution (min-norm), shape (3,)
v1      = None   # null-space basis vector 1, shape (3,)
v2      = None   # null-space basis vector 2, shape (3,)

# Constrained optimum
theta_opt = None   # shape (3,)
J_opt     = None
s_opt     = None   # (s, t) coords of optimum on constraint plane
t_opt     = None

# Cached contour grid
_S_grid  = None
_T_grid  = None
_J_grid  = None
_s_range = None   # (s_lo, s_hi)
_t_range = None   # (t_lo, t_hi)

hover_s    = None
hover_t    = None
selected_s = None
selected_t = None

scatter_xlim = None
scatter_ylim = None


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def poly_eval(theta, x):
    """Evaluate a0 + a1*x + a2*x² for an array x."""
    return theta[0] + theta[1] * x + theta[2] * x ** 2


def design_matrix(x):
    """Return N×3 design matrix [1, x, x²]."""
    return np.column_stack([np.ones_like(x), x, x ** 2])


def objective_theta(theta):
    """J(θ) = ‖A·θ − y‖²."""
    return float(np.sum((design_matrix(x_data) @ theta - y_data) ** 2))


def objective_st(s, t):
    """J evaluated at θ_p + s·v1 + t·v2 (a point on the constraint plane)."""
    return objective_theta(theta_p + s * v1 + t * v2)


def _build_constraint_plane():
    """Compute θ_p (min-norm particular solution) and null-space basis {v1, v2}."""
    global theta_p, v1, v2
    norm_sq = float(c_val @ c_val)
    if norm_sq < 1e-14:
        theta_p = v1 = v2 = None
        return
    # Minimum-norm particular solution: C†·d = Cᵀ·(CCᵀ)⁻¹·d
    theta_p = c_val * (d_val / norm_sq)
    # Null space via SVD of C (1×3 matrix)
    _, _, Vt = np.linalg.svd(c_val.reshape(1, 3))
    v1 = Vt[1]
    v2 = Vt[2]


def compute_constrained_opt():
    """Solve min ‖Aθ−y‖² s.t. C·θ = d via the KKT system, then map to (s,t)."""
    global theta_opt, J_opt, s_opt, t_opt
    if x_data is None or theta_p is None:
        theta_opt = J_opt = s_opt = t_opt = None
        return
    A   = design_matrix(x_data)
    AtA = A.T @ A
    Aty = A.T @ y_data
    C   = c_val.reshape(1, 3)
    KKT = np.block([[AtA, C.T], [C, np.zeros((1, 1))]])
    rhs = np.append(Aty, d_val)
    try:
        sol       = np.linalg.solve(KKT, rhs)
        theta_opt = sol[:3]
        J_opt     = objective_theta(theta_opt)
        delta     = theta_opt - theta_p
        s_opt     = float(v1 @ delta)
        t_opt     = float(v2 @ delta)
    except np.linalg.LinAlgError:
        theta_opt = J_opt = s_opt = t_opt = None


def _build_grid():
    """Build (s, t) grid and evaluate J on the constraint plane (vectorised)."""
    global _S_grid, _T_grid, _J_grid, _s_range, _t_range
    if theta_p is None or v1 is None or v2 is None:
        _S_grid = _T_grid = _J_grid = _s_range = _t_range = None
        return

    # Centre grid on constrained optimum with generous margin
    sc   = s_opt if s_opt is not None else 0.0
    tc   = t_opt if t_opt is not None else 0.0
    half = max(GRID_HALF, 2.0 * max(abs(sc), abs(tc), 1.0))

    s_vals = np.linspace(sc - half, sc + half, GRID_RES)
    t_vals = np.linspace(tc - half, tc + half, GRID_RES)
    S, T   = np.meshgrid(s_vals, t_vals)

    # Expand quadratic: J(s,t) = ‖r + s·e1 + t·e2‖²
    A  = design_matrix(x_data)
    r  = A @ theta_p - y_data   # residual at particular solution
    e1 = A @ v1
    e2 = A @ v2

    rr   = np.dot(r,  r)
    re1  = np.dot(r,  e1)
    re2  = np.dot(r,  e2)
    e1e1 = np.dot(e1, e1)
    e1e2 = np.dot(e1, e2)
    e2e2 = np.dot(e2, e2)

    _J_grid = (rr
               + 2*S*re1   + 2*T*re2
               + S**2*e1e1 + 2*S*T*e1e2 + T**2*e2e2)
    _S_grid, _T_grid = S, T
    _s_range = (sc - half, sc + half)
    _t_range = (tc - half, tc + half)


def _in_grid_bounds(s, t):
    return (_s_range is not None
            and _s_range[0] <= s <= _s_range[1]
            and _t_range[0] <= t <= _t_range[1])


# ---------------------------------------------------------------------------
# Event callbacks
# ---------------------------------------------------------------------------
def generate_new_points(event=None):
    global x_data, y_data, selected_s, selected_t, hover_s, hover_t
    global scatter_xlim, scatter_ylim

    a0_t = np.random.uniform(-1.0,  1.0)
    a1_t = np.random.uniform(-2.0,  2.0)
    a2_t = np.random.uniform(-0.8,  0.8)
    x_data = np.random.uniform(-3.0, 3.0, N)
    y_data = (a0_t + a1_t * x_data + a2_t * x_data ** 2
              + np.random.randn(N) * 0.5)

    _build_constraint_plane()
    compute_constrained_opt()
    _build_grid()

    xpad = 0.5
    xmin = float(x_data.min()) - xpad
    xmax = float(x_data.max()) + xpad
    ypad = max(1.0, 0.35 * float(y_data.ptp()))
    scatter_xlim = (xmin, xmax)
    scatter_ylim = (float(y_data.min()) - ypad, float(y_data.max()) + ypad)

    selected_s = selected_t = hover_s = hover_t = None
    redraw()


def _update_constraint(new_c=None, new_d=None):
    global c_val, d_val, selected_s, selected_t, hover_s, hover_t
    if new_c is not None:
        c_val = new_c
    if new_d is not None:
        d_val = new_d
    selected_s = selected_t = hover_s = hover_t = None
    _build_constraint_plane()
    compute_constrained_opt()
    _build_grid()
    redraw()


def on_c1_submit(text):
    try:
        nc = c_val.copy(); nc[0] = float(text)
        _update_constraint(new_c=nc)
    except ValueError:
        pass


def on_c2_submit(text):
    try:
        nc = c_val.copy(); nc[1] = float(text)
        _update_constraint(new_c=nc)
    except ValueError:
        pass


def on_c3_submit(text):
    try:
        nc = c_val.copy(); nc[2] = float(text)
        _update_constraint(new_c=nc)
    except ValueError:
        pass


def on_d_submit(text):
    try:
        _update_constraint(new_d=float(text))
    except ValueError:
        pass


def on_hover(event):
    global hover_s, hover_t
    if event.inaxes is not ax_contour or x_data is None or _s_range is None:
        if hover_s is not None:
            hover_s = hover_t = None
            redraw()
        return
    s, t = event.xdata, event.ydata
    if not _in_grid_bounds(s, t):
        if hover_s is not None:
            hover_s = hover_t = None
            redraw()
        return
    hover_s, hover_t = s, t
    redraw()


def on_click(event):
    global selected_s, selected_t
    if event.inaxes is not ax_contour or x_data is None or _s_range is None:
        return
    s, t = event.xdata, event.ydata
    if not _in_grid_bounds(s, t):
        return
    selected_s, selected_t = s, t
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
    ax_scatter.cla()
    xline = np.linspace(scatter_xlim[0], scatter_xlim[1], 400)

    ax_scatter.scatter(x_data, y_data, color='#333333', s=35, zorder=4,
                       label='Data')

    if hover_s is not None:
        th = theta_p + hover_s * v1 + hover_t * v2
        ax_scatter.plot(xline, poly_eval(th, xline),
                        color=HOV_COLOR, lw=1.5, ls='--', zorder=2,
                        alpha=0.7, label='Preview')

    if selected_s is not None:
        th = theta_p + selected_s * v1 + selected_t * v2
        ax_scatter.plot(xline, poly_eval(th, xline),
                        color=SEL_COLOR, lw=2.0, ls='--', zorder=3,
                        label='Selected')

    if theta_opt is not None:
        ax_scatter.plot(xline, poly_eval(theta_opt, xline),
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
    ax_scatter.set_title('Data & Fitted Polynomial', fontsize=13, pad=8)
    ax_scatter.legend(fontsize=10, loc='best')


def draw_contour():
    ax_contour.cla()

    if _J_grid is not None:
        ax_contour.contourf(_S_grid, _T_grid, _J_grid,
                            levels=30, cmap='RdYlBu_r', alpha=0.55)
        ax_contour.contour(_S_grid, _T_grid, _J_grid,
                           levels=20, cmap='RdYlBu_r', linewidths=0.9)

    # Cost-level ellipse for hover
    if _J_grid is not None and hover_s is not None:
        Jh = objective_st(hover_s, hover_t)
        ax_contour.contour(_S_grid, _T_grid, _J_grid,
                           levels=[Jh], colors=[HOV_COLOR],
                           linewidths=1.4, linestyles=':', alpha=0.85, zorder=5)

    # Cost-level ellipse for selected
    if _J_grid is not None and selected_s is not None:
        Js = objective_st(selected_s, selected_t)
        ax_contour.contour(_S_grid, _T_grid, _J_grid,
                           levels=[Js], colors=[SEL_COLOR],
                           linewidths=1.4, linestyles=':', alpha=0.85, zorder=5)

    # Minimum-cost ellipse (constrained optimum)
    if _J_grid is not None and J_opt is not None:
        ax_contour.contour(_S_grid, _T_grid, _J_grid,
                           levels=[J_opt], colors=['tab:red'],
                           linewidths=2.0, zorder=6)

    # --- markers with drop-lines ---
    def _drop_lines(ax, sx, ty, color, lw, alpha):
        ax.plot([sx, sx], [_t_range[0], ty],
                color=color, lw=lw, ls=':', alpha=alpha, zorder=7)
        ax.plot([_s_range[0], sx], [ty, ty],
                color=color, lw=lw, ls=':', alpha=alpha, zorder=7)

    if s_opt is not None:
        _drop_lines(ax_contour, s_opt, t_opt, OPT_COLOR, 1.0, 0.9)
        ax_contour.plot(s_opt, t_opt,
                        marker='*', color=OPT_COLOR, markersize=14, zorder=8,
                        markeredgecolor='#444444', markeredgewidth=0.5,
                        label='Constrained opt.', linestyle='None')

    if hover_s is not None:
        _drop_lines(ax_contour, hover_s, hover_t, HOV_COLOR, 0.7, 0.45)
        ax_contour.plot(hover_s, hover_t,
                        marker='o', color=HOV_COLOR, markersize=7, zorder=8,
                        markeredgecolor='#555555', markeredgewidth=0.5,
                        alpha=0.85, label='Preview', linestyle='None')

    if selected_s is not None:
        _drop_lines(ax_contour, selected_s, selected_t, SEL_COLOR, 0.7, 0.5)
        ax_contour.plot(selected_s, selected_t,
                        marker='o', color=SEL_COLOR, markersize=9, zorder=8,
                        markeredgecolor='#333333', markeredgewidth=0.5,
                        label='Selected', linestyle='None')

    if _s_range:
        ax_contour.set_xlim(*_s_range)
        ax_contour.set_ylim(*_t_range)
    else:
        ax_contour.set_xlim(-GRID_HALF, GRID_HALF)
        ax_contour.set_ylim(-GRID_HALF, GRID_HALF)

    ax_contour.set_aspect('equal', adjustable='box')
    ax_contour.set_xlabel('$s$', fontsize=13)
    ax_contour.set_ylabel('$t$', fontsize=13)
    ax_contour.set_title(
        'Objective $J(s,\\,t)$ on Constraint Plane\n'
        r'$\theta(s,t)=\theta_p + s\,\mathbf{v}_1 + t\,\mathbf{v}_2$',
        fontsize=12, pad=6)
    ax_contour.legend(fontsize=9, loc='upper right')


def draw_text():
    ax_text.cla()
    ax_text.axis('off')
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    if x_data is None:
        return

    c1, c2, c3 = c_val

    # Row 1 — constraint equation
    c_str = (f"Constraint:  "
             f"${c1:g}\\cdot a_0 "
             f"+ {c2:g}\\cdot a_1 "
             f"+ {c3:g}\\cdot a_2 = {d_val:g}$")
    ax_text.text(0.01, 0.82, c_str, fontsize=11, color='#222222',
                 va='center', transform=ax_text.transAxes)

    # Row 2 — constrained optimum
    if theta_opt is not None:
        opt_str = (f"Constrained opt.:  "
                   f"$a_0^* = {theta_opt[0]:+.3f}$,   "
                   f"$a_1^* = {theta_opt[1]:+.3f}$,   "
                   f"$a_2^* = {theta_opt[2]:+.3f}$,   "
                   f"$J^* = {J_opt:.3f}$")
        ax_text.text(0.01, 0.57, opt_str, fontsize=11,
                     color=OPT_COLOR, fontweight='bold',
                     va='center', transform=ax_text.transAxes)
    elif theta_p is None:
        ax_text.text(0.01, 0.57,
                     'Degenerate constraint  ($c_1 = c_2 = c_3 = 0$).',
                     fontsize=11, color='tab:red', va='center',
                     transform=ax_text.transAxes)
    else:
        ax_text.text(0.01, 0.57,
                     'Constrained optimum unavailable (singular KKT system).',
                     fontsize=11, color='tab:red', va='center',
                     transform=ax_text.transAxes)

    # Row 3 — hover / selected / hint
    if selected_s is not None:
        th = theta_p + selected_s * v1 + selected_t * v2
        Js = objective_theta(th)
        sel_str = (f"Selected:  "
                   f"$a_0 = {th[0]:+.3f}$,   "
                   f"$a_1 = {th[1]:+.3f}$,   "
                   f"$a_2 = {th[2]:+.3f}$,   "
                   f"$J = {Js:.3f}$")
        ax_text.text(0.01, 0.24, sel_str, fontsize=11, color=SEL_COLOR,
                     va='center', transform=ax_text.transAxes)
    elif hover_s is not None:
        th = theta_p + hover_s * v1 + hover_t * v2
        Jh = objective_theta(th)
        hov_str = (f"Preview:  "
                   f"$a_0 = {th[0]:+.3f}$,   "
                   f"$a_1 = {th[1]:+.3f}$,   "
                   f"$a_2 = {th[2]:+.3f}$,   "
                   f"$J = {Jh:.3f}$")
        ax_text.text(0.01, 0.24, hov_str, fontsize=11, color=HOV_COLOR,
                     va='center', transform=ax_text.transAxes)
    else:
        ax_text.text(0.01, 0.24,
                     'Hover over the contour to preview a constrained solution.',
                     fontsize=10, color='#999999', style='italic',
                     va='center', transform=ax_text.transAxes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig = plt.figure(figsize=(15, 8.5))
    fig.canvas.manager.set_window_title(
        'ALADA – Constrained Least Squares: 2nd-Order Polynomial')

    # ---- main axes ----
    ax_scatter = fig.add_axes([0.05, 0.24, 0.36, 0.70])
    ax_contour = fig.add_axes([0.50, 0.24, 0.44, 0.70])
    ax_text    = fig.add_axes([0.05, 0.01, 0.92, 0.14])
    ax_text.axis('off')

    # ---- Generate button (below scatter) ----
    ax_btn = fig.add_axes([0.09, 0.118, 0.10, 0.062])
    btn = Button(ax_btn, 'Generate\nPoints', color='#d0e8f5', hovercolor='#a0c8ef')
    btn.on_clicked(generate_new_points)

    # ---- Constraint controls (below contour, x: 0.50 → 0.97) ----
    fig.text(0.505, 0.150, 'Constraint:', fontsize=11,
             va='center', ha='left', color='#222222')

    # c1 box
    ax_c1 = fig.add_axes([0.593, 0.118, 0.044, 0.052])
    tb_c1 = TextBox(ax_c1, '', initial='1.0', textalignment='center')
    tb_c1.on_submit(on_c1_submit)
    fig.text(0.615, 0.178, '$c_1$', fontsize=12,
             va='bottom', ha='center', color='#555555')
    fig.text(0.641, 0.150, r'$\cdot a_0\;+$', fontsize=11,
             va='center', ha='left', color='#222222')

    # c2 box
    ax_c2 = fig.add_axes([0.700, 0.118, 0.044, 0.052])
    tb_c2 = TextBox(ax_c2, '', initial='0.0', textalignment='center')
    tb_c2.on_submit(on_c2_submit)
    fig.text(0.722, 0.178, '$c_2$', fontsize=12,
             va='bottom', ha='center', color='#555555')
    fig.text(0.748, 0.150, r'$\cdot a_1\;+$', fontsize=11,
             va='center', ha='left', color='#222222')

    # c3 box
    ax_c3 = fig.add_axes([0.807, 0.118, 0.044, 0.052])
    tb_c3 = TextBox(ax_c3, '', initial='0.0', textalignment='center')
    tb_c3.on_submit(on_c3_submit)
    fig.text(0.829, 0.178, '$c_3$', fontsize=12,
             va='bottom', ha='center', color='#555555')
    fig.text(0.855, 0.150, r'$\cdot a_2\;=$', fontsize=11,
             va='center', ha='left', color='#222222')

    # d box
    ax_d = fig.add_axes([0.913, 0.118, 0.044, 0.052])
    tb_d = TextBox(ax_d, '', initial='0.0', textalignment='center')
    tb_d.on_submit(on_d_submit)
    fig.text(0.935, 0.178, '$d$', fontsize=12,
             va='bottom', ha='center', color='#555555')

    # ---- event connections ----
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    generate_new_points()
    plt.show()
