"""
3D parameter-space visualisation for constrained least-squares polynomial fitting.

Polynomial model : y = a₀ + a₁x + a₂x²
Constraint       : c₁·a₀ + c₂·a₁ + c₃·a₂ = d

In (a₀, a₁, a₂) parameter space:
  · J(θ)     is a bowl whose level sets are ellipsoids centred on θ̂
              (the unconstrained minimum).
  · C·θ = d  is a 2-D plane slicing through that space.
  · The constrained optimum θ* is the point on the plane where the
    *smallest* ellipsoid (J = J*) just becomes tangent to the plane.

Panels
------
  Left  : data scatter with polynomial fits (2-D).
  Right : 3-D parameter space –
            • semi-transparent point cloud coloured by J(θ)
            • constraint plane  (blue, semi-transparent)
            • tangent ellipsoid J = J* (green, semi-transparent)
            • unconstrained optimum θ̂ (dark-blue dot)
            • constrained optimum   θ* (green star)
            • hover preview point    (grey dot on the plane)
            • selected point         (red star on the plane)

Controls
--------
  "Generate Points" button   : new random data, reset everything.
  c1 / c2 / c3 / d text boxes: update constraint C·θ = d (press Enter).
  Hover over constraint plane : preview the corresponding polynomial.
  Click on constraint plane   : lock a selected solution (short click only,
                                 not drag — drag still rotates the view).
  "Reset View"   button       : restore default camera angle.
  Escape                      : close the window.

Ray-casting
-----------
  A click at display pixel (x, y) defines a viewing ray in 3-D data space
  via:  p = inv(get_proj()) @ [x_proj, y_proj, z_clip, 1]
  The ray is intersected with the constraint plane C·θ = d analytically.
  Left-click selects; hover previews (drag is distinguished by a 5-px
  movement threshold).

Author: Sivakumar Balasubramanian
Date: 14 April 2026
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D      # noqa – registers projection
from mpl_toolkits.mplot3d import proj3d      # forward projection utility
from matplotlib.widgets import Button, TextBox
import platform

# ---------------------------------------------------------------------------
# Font / toolbar
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
N            = 20
PC_N         = 13         # point-cloud samples per axis  (13³ ≈ 2200 pts)
PICK_TOL_PX  = 20         # pixel radius for nearest-point plane picking
ELLIPS_PTS   = 36         # resolution of ellipsoid mesh
PLANE_PTS    = 18         # resolution of constraint-plane mesh
CLICK_TOL_PX = 5          # pixel-distance threshold: click vs. drag

OPT_COLOR  = 'tab:green'
UCON_COLOR = '#222288'    # unconstrained optimum colour
HOV_COLOR  = '#888888'    # hover preview colour
SEL_COLOR  = 'tab:red'    # selected point colour
PC_CMAP    = 'RdYlBu_r'  # point-cloud colour map

DEFAULT_ELEV = 22.
DEFAULT_AZIM = 40.

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
x_data = None
y_data = None

c_val = np.array([1., 0., 0.])
d_val = 0.

# Constraint-plane basis
theta_p = None
v1      = None
v2      = None

# Optima
theta_hat = None     # unconstrained minimum
J_min_unc = None
theta_opt = None     # constrained minimum
J_opt     = None

# Interactive selection
hover_pt    = None   # 3-D point on constraint plane (hover)
selected_pt = None   # 3-D point on constraint plane (locked)

# Mouse-state for distinguishing click from drag
_press_pos      = None   # (x, y) display pixels at mouse-press
_mouse_btn_down = False

scatter_xlim = None
scatter_ylim = None

# Cached point cloud  (a0, a1, a2, J) arrays
_pc_a0 = _pc_a1 = _pc_a2 = _pc_J = None
_pc_center = None
_pc_half   = None


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def design_matrix(x):
    return np.column_stack([np.ones_like(x), x, x ** 2])


def poly_eval(theta, x):
    return theta[0] + theta[1] * x + theta[2] * x ** 2


def objective_theta(theta):
    return float(np.sum((design_matrix(x_data) @ theta - y_data) ** 2))


def _build_constraint_plane():
    global theta_p, v1, v2
    norm_sq = float(c_val @ c_val)
    if norm_sq < 1e-14:
        theta_p = v1 = v2 = None
        return
    theta_p = c_val * (d_val / norm_sq)
    _, _, Vt = np.linalg.svd(c_val.reshape(1, 3))
    v1, v2 = Vt[1], Vt[2]


def _compute_unconstrained_opt():
    global theta_hat, J_min_unc
    if x_data is None:
        theta_hat = J_min_unc = None
        return
    A = design_matrix(x_data)
    try:
        theta_hat = np.linalg.solve(A.T @ A, A.T @ y_data)
        J_min_unc = objective_theta(theta_hat)
    except np.linalg.LinAlgError:
        theta_hat = J_min_unc = None


def compute_constrained_opt():
    global theta_opt, J_opt
    if x_data is None or theta_p is None:
        theta_opt = J_opt = None
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
    except np.linalg.LinAlgError:
        theta_opt = J_opt = None


def _half_range():
    """Adaptive half-range for the 3-D display, based on ellipsoid size."""
    if theta_hat is None:
        return 4.
    base = 4.
    if J_opt is not None and J_min_unc is not None and J_opt > J_min_unc + 1e-6:
        A = design_matrix(x_data)
        H = A.T @ A
        lam = np.linalg.eigvalsh(H)
        lam_min = max(lam.min(), 1e-8)
        max_sa = np.sqrt((J_opt - J_min_unc) / lam_min)
        base = max(base, 2.5 * max_sa)
    if theta_opt is not None:
        dist = float(np.max(np.abs(theta_opt - theta_hat)))
        base = max(base, dist * 1.4 + 1.)
    return float(base)


# ---------------------------------------------------------------------------
# Picking: display pixel → nearest point on constraint plane
# ---------------------------------------------------------------------------
def _nearest_plane_point(event):
    """
    Find the point on the rendered constraint-plane patch whose projected
    screen position is closest to the mouse event pixel.

    Uses only the *forward* projection (proj3d.proj_transform), which is
    known to be reliable for matplotlib 3-D axes.  The view→display mapping
    is derived from the 8 corners of the axis volume, so it adapts to any
    camera angle without needing transData.inverted().

    Returns the 3-D parameter point (ndarray shape (3,)), or None if the
    cursor is more than PICK_TOL_PX pixels away from every plane sample.
    """
    if theta_p is None or v1 is None or x_data is None:
        return None
    if event.x is None or event.y is None:
        return None

    # 1. Sample a fine grid on the constraint plane
    h   = _half_range()
    ctr = theta_opt if theta_opt is not None else theta_p
    n   = 50                          # 50×50 = 2500 samples
    sv  = np.linspace(-h, h, n)
    tv  = np.linspace(-h, h, n)
    S, T = np.meshgrid(sv, tv)
    pts  = (ctr[:, None, None]
            + S[None] * v1[:, None, None]
            + T[None] * v2[:, None, None])
    a0s = pts[0].ravel()
    a1s = pts[1].ravel()
    a2s = pts[2].ravel()

    # 2. Forward-project all plane samples to view coordinates
    M = ax_3d.get_proj()
    xs2d, ys2d, _ = proj3d.proj_transform(a0s, a1s, a2s, M)

    # 3. Estimate the view→display mapping from the 8 axis-volume corners
    #    (avoids transData.inverted() which is unreliable on 3-D axes)
    xlim = ax_3d.get_xlim3d()
    ylim = ax_3d.get_ylim3d()
    zlim = ax_3d.get_zlim3d()
    corners = np.array([[x, y, z]
                        for x in xlim for y in ylim for z in zlim])
    cxv, cyv, _ = proj3d.proj_transform(
        corners[:, 0], corners[:, 1], corners[:, 2], M)
    xv_min, xv_max = float(cxv.min()), float(cxv.max())
    yv_min, yv_max = float(cyv.min()), float(cyv.max())
    if xv_max <= xv_min or yv_max <= yv_min:
        return None

    ax_bbox = ax_3d.get_window_extent()
    xd = ax_bbox.x0 + (xs2d - xv_min) / (xv_max - xv_min) * ax_bbox.width
    yd = ax_bbox.y0 + (ys2d - yv_min) / (yv_max - yv_min) * ax_bbox.height

    # 4. Pick the sample nearest to the click/hover pixel
    dist2 = (xd - event.x) ** 2 + (yd - event.y) ** 2
    idx   = int(np.argmin(dist2))
    if dist2[idx] > PICK_TOL_PX ** 2:
        return None

    return np.array([a0s[idx], a1s[idx], a2s[idx]])


# ---------------------------------------------------------------------------
# Build cached geometry
# ---------------------------------------------------------------------------
def _build_point_cloud():
    """Vectorised: evaluate J on an (n³) grid in parameter space."""
    global _pc_a0, _pc_a1, _pc_a2, _pc_J, _pc_center, _pc_half
    if x_data is None or theta_hat is None:
        _pc_a0 = _pc_a1 = _pc_a2 = _pc_J = None
        return
    h  = _half_range()
    c  = theta_hat
    n  = PC_N

    a0v = np.linspace(c[0] - h,     c[0] + h,     n)
    a1v = np.linspace(c[1] - h,     c[1] + h,     n)
    a2v = np.linspace(c[2] - h*0.6, c[2] + h*0.6, n)
    A0, A1, A2 = np.meshgrid(a0v, a1v, a2v, indexing='ij')

    thetas = np.stack([A0.ravel(), A1.ravel(), A2.ravel()], axis=1)

    Dm    = design_matrix(x_data)
    preds = thetas @ Dm.T
    Jvals = np.sum((preds - y_data[None, :]) ** 2, axis=1)

    _pc_a0, _pc_a1, _pc_a2, _pc_J = (A0.ravel(), A1.ravel(),
                                       A2.ravel(), Jvals)
    _pc_center = c
    _pc_half   = h


def _ellipsoid_surface():
    """
    Mesh for  (θ−θ̂)ᵀ H (θ−θ̂) = J* − J_min  (tangent ellipsoid).
    Returns (ex, ey, ez) or (None, None, None).
    """
    if theta_hat is None or J_opt is None or J_min_unc is None:
        return None, None, None
    gap = J_opt - J_min_unc
    if gap < 1e-10:
        return None, None, None

    A = design_matrix(x_data)
    H = A.T @ A
    lam, Q = np.linalg.eigh(H)
    lam    = np.maximum(lam, 1e-10)
    scale  = np.sqrt(gap / lam)

    n = ELLIPS_PTS
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi,     n // 2)
    U, V = np.meshgrid(u, v)

    sx = (np.sin(V) * np.cos(U)).ravel()
    sy = (np.sin(V) * np.sin(U)).ravel()
    sz = np.cos(V).ravel()

    pts = Q @ (scale[:, None] * np.stack([sx, sy, sz])) + theta_hat[:, None]
    return pts[0].reshape(V.shape), pts[1].reshape(V.shape), pts[2].reshape(V.shape)


def _constraint_plane_surface():
    """
    Mesh for the constraint plane patch, centred on θ* (or θ_p).
    Returns (px, py, pz) or (None, None, None).
    """
    if theta_p is None or v1 is None or v2 is None:
        return None, None, None
    ctr = theta_opt if theta_opt is not None else theta_p
    h   = _half_range()
    n   = PLANE_PTS

    s_v = np.linspace(-h, h, n)
    t_v = np.linspace(-h, h, n)
    S, T = np.meshgrid(s_v, t_v)

    pts = (ctr[:, None, None]
           + S[None] * v1[:, None, None]
           + T[None] * v2[:, None, None])
    return pts[0], pts[1], pts[2]


# ---------------------------------------------------------------------------
# Event callbacks
# ---------------------------------------------------------------------------
def generate_new_points(event=None):
    global x_data, y_data, scatter_xlim, scatter_ylim
    global hover_pt, selected_pt

    a0_t = np.random.uniform(-1.0,  1.0)
    a1_t = np.random.uniform(-2.0,  2.0)
    a2_t = np.random.uniform(-0.8,  0.8)
    x_data = np.random.uniform(-3., 3., N)
    y_data = (a0_t + a1_t * x_data + a2_t * x_data ** 2
              + np.random.randn(N) * 0.5)

    _build_constraint_plane()
    _compute_unconstrained_opt()
    compute_constrained_opt()
    _build_point_cloud()

    xpad = 0.5
    xmin = float(x_data.min()) - xpad
    xmax = float(x_data.max()) + xpad
    ypad = max(1., 0.35 * float(y_data.ptp()))
    scatter_xlim = (xmin, xmax)
    scatter_ylim = (float(y_data.min()) - ypad, float(y_data.max()) + ypad)

    hover_pt = selected_pt = None
    redraw()


def _update_constraint(new_c=None, new_d=None):
    global c_val, d_val, hover_pt, selected_pt
    if new_c is not None:
        c_val = new_c
    if new_d is not None:
        d_val = new_d
    hover_pt = selected_pt = None
    _build_constraint_plane()
    compute_constrained_opt()
    _build_point_cloud()
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


def on_reset_view(event=None):
    ax_3d.view_init(elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)
    fig.canvas.draw_idle()


def on_key(event):
    if event.key == 'escape':
        plt.close(fig)


# ---------------------------------------------------------------------------
# Mouse interaction: hover preview + click-to-select on constraint plane
# ---------------------------------------------------------------------------
def on_mouse_press(event):
    global _press_pos, _mouse_btn_down
    if event.button == 1 and event.inaxes is ax_3d:
        _press_pos      = (event.x, event.y)
        _mouse_btn_down = True


def on_mouse_release(event):
    """Select a point on the plane if the release is close to the press (click)."""
    global selected_pt, _press_pos, _mouse_btn_down
    _mouse_btn_down = False
    if event.button != 1 or event.inaxes is not ax_3d or _press_pos is None:
        _press_pos = None
        return

    dx = (event.x or 0) - _press_pos[0]
    dy = (event.y or 0) - _press_pos[1]
    _press_pos = None

    if dx ** 2 + dy ** 2 > CLICK_TOL_PX ** 2:
        return   # was a rotation drag, not a click

    pt = _nearest_plane_point(event)
    if pt is not None:
        selected_pt = pt
        redraw()


def on_hover(event):
    """Preview the polynomial for the point on the constraint plane under the cursor."""
    global hover_pt
    if _mouse_btn_down:          # suppress during rotation drag
        return
    if event.inaxes is not ax_3d or theta_p is None:
        if hover_pt is not None:
            hover_pt = None
            redraw()
        return

    pt = _nearest_plane_point(event)
    if pt is None:
        if hover_pt is not None:
            hover_pt = None
            redraw()
        return

    hover_pt = pt
    redraw()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def redraw():
    if x_data is None:
        return
    draw_scatter()
    draw_3d()
    draw_text()
    fig.canvas.draw_idle()


def draw_scatter():
    ax_scatter.cla()
    xline = np.linspace(scatter_xlim[0], scatter_xlim[1], 400)

    ax_scatter.scatter(x_data, y_data, color='#333333', s=35, zorder=4,
                       label='Data')

    # Hover preview (below selected)
    if hover_pt is not None:
        ax_scatter.plot(xline, poly_eval(hover_pt, xline),
                        color=HOV_COLOR, lw=1.5, ls='--', zorder=3,
                        alpha=0.8, label='Preview')

    # Locked selection
    if selected_pt is not None:
        ax_scatter.plot(xline, poly_eval(selected_pt, xline),
                        color=SEL_COLOR, lw=2.0, ls='--', zorder=4,
                        label='Selected')

    # Unconstrained optimum
    if theta_hat is not None:
        ax_scatter.plot(xline, poly_eval(theta_hat, xline),
                        color=UCON_COLOR, lw=1.8, ls='--', zorder=5,
                        label='Unconstrained opt. $\\hat{\\theta}$')

    # Constrained optimum (on top)
    if theta_opt is not None:
        ax_scatter.plot(xline, poly_eval(theta_opt, xline),
                        color=OPT_COLOR, lw=2.5, zorder=6,
                        label='Constrained opt.   $\\theta^*$')

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


def draw_3d():
    # Preserve the user's current viewing angle
    try:
        elev, azim = ax_3d.elev, ax_3d.azim
    except Exception:
        elev, azim = DEFAULT_ELEV, DEFAULT_AZIM

    ax_3d.cla()

    h = _half_range() if theta_hat is not None else 4.

    # ------------------------------------------------------------------
    # 1. Point cloud
    # ------------------------------------------------------------------
    if _pc_J is not None:
        J_show_max = J_opt * 4.0 if J_opt is not None else float(_pc_J.max())
        mask  = _pc_J <= J_show_max
        J_sub = _pc_J[mask]
        if J_sub.size > 0:
            J_norm = (J_sub - J_sub.min()) / max(J_sub.ptp(), 1e-6)
            ax_3d.scatter(_pc_a0[mask], _pc_a1[mask], _pc_a2[mask],
                          c=J_norm, cmap=PC_CMAP,
                          s=8, alpha=0.08, linewidths=0, depthshade=True,
                          zorder=1)

    # ------------------------------------------------------------------
    # 2. Constraint plane
    # ------------------------------------------------------------------
    px, py, pz = _constraint_plane_surface()
    if px is not None:
        ax_3d.plot_surface(px, py, pz,
                           color='steelblue', alpha=0.18,
                           linewidth=0, antialiased=True, zorder=2)
        ax_3d.plot_wireframe(px, py, pz,
                             color='steelblue', alpha=0.30,
                             linewidth=0.4, rstride=3, cstride=3, zorder=3)

    # ------------------------------------------------------------------
    # 3. Tangent ellipsoid  J = J*
    # ------------------------------------------------------------------
    ex, ey, ez = _ellipsoid_surface()
    if ex is not None:
        ax_3d.plot_surface(ex, ey, ez,
                           color=OPT_COLOR, alpha=0.22,
                           linewidth=0, antialiased=True, zorder=4)
        ax_3d.plot_wireframe(ex, ey, ez,
                             color=OPT_COLOR, alpha=0.35,
                             linewidth=0.4, rstride=3, cstride=3, zorder=5)

    # ------------------------------------------------------------------
    # 4. Unconstrained optimum θ̂
    # ------------------------------------------------------------------
    if theta_hat is not None:
        ax_3d.scatter(*theta_hat, s=80, color=UCON_COLOR,
                      marker='o', zorder=9, depthshade=False,
                      label=r'Unconstrained opt. $\hat{\theta}$')

    # ------------------------------------------------------------------
    # 5. Constrained optimum θ*
    # ------------------------------------------------------------------
    if theta_opt is not None:
        ax_3d.scatter(*theta_opt, s=160, color=OPT_COLOR,
                      marker='*', zorder=10, depthshade=False,
                      edgecolors='#333333', linewidths=0.6,
                      label=r'Constrained opt. $\theta^*$')
        if _pc_half is not None:
            z_floor = _pc_center[2] - _pc_half * 0.6
            ax_3d.plot([theta_opt[0], theta_opt[0]],
                       [theta_opt[1], theta_opt[1]],
                       [z_floor, theta_opt[2]],
                       color=OPT_COLOR, lw=0.9, ls=':', alpha=0.5, zorder=6)

    # ------------------------------------------------------------------
    # 6. Hover preview point on constraint plane
    # ------------------------------------------------------------------
    if hover_pt is not None:
        ax_3d.scatter(*hover_pt, s=80, color=HOV_COLOR,
                      marker='o', zorder=11, depthshade=False,
                      alpha=0.85, edgecolors='#555555', linewidths=0.5,
                      label='Preview')

    # ------------------------------------------------------------------
    # 7. Selected point on constraint plane
    # ------------------------------------------------------------------
    if selected_pt is not None:
        ax_3d.scatter(*selected_pt, s=140, color=SEL_COLOR,
                      marker='*', zorder=12, depthshade=False,
                      edgecolors='#333333', linewidths=0.6,
                      label='Selected')

    # ------------------------------------------------------------------
    # Axes cosmetics
    # ------------------------------------------------------------------
    ctr = theta_hat if theta_hat is not None else np.zeros(3)
    ax_3d.set_xlim(ctr[0] - h,     ctr[0] + h)
    ax_3d.set_ylim(ctr[1] - h,     ctr[1] + h)
    ax_3d.set_zlim(ctr[2] - h*0.6, ctr[2] + h*0.6)

    ax_3d.set_xlabel('$a_0$', fontsize=12, labelpad=6)
    ax_3d.set_ylabel('$a_1$', fontsize=12, labelpad=6)
    ax_3d.set_zlabel('$a_2$', fontsize=12, labelpad=6)
    ax_3d.set_title(
        'Parameter Space  $(a_0,\\,a_1,\\,a_2)$\n'
        'Ellipsoid: $J=J^*$  ·  Plane: $C\\theta=d$',
        fontsize=11, pad=10)
    ax_3d.legend(fontsize=9, loc='upper left')
    ax_3d.view_init(elev=elev, azim=azim)


def draw_text():
    ax_text.cla()
    ax_text.axis('off')
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    if x_data is None:
        return

    c1, c2, c3 = c_val

    # Row 1 – constraint equation
    c_str = (f"Constraint:  "
             f"${c1:g}\\cdot a_0 + {c2:g}\\cdot a_1 + {c3:g}\\cdot a_2 = {d_val:g}$")
    ax_text.text(0.01, 0.88, c_str, fontsize=11, color='#222222',
                 va='center', transform=ax_text.transAxes)

    # Row 2 – unconstrained optimum
    if theta_hat is not None:
        ustr = (f"Unconstrained opt.: "
                f"$\\hat{{a}}_0={theta_hat[0]:+.3f}$,  "
                f"$\\hat{{a}}_1={theta_hat[1]:+.3f}$,  "
                f"$\\hat{{a}}_2={theta_hat[2]:+.3f}$,  "
                f"$J_{{\\min}}={J_min_unc:.3f}$")
        ax_text.text(0.01, 0.63, ustr, fontsize=11, color=UCON_COLOR,
                     va='center', transform=ax_text.transAxes)

    # Row 3 – constrained optimum
    if theta_opt is not None:
        ostr = (f"Constrained opt.:   "
                f"$a_0^*={theta_opt[0]:+.3f}$,  "
                f"$a_1^*={theta_opt[1]:+.3f}$,  "
                f"$a_2^*={theta_opt[2]:+.3f}$,  "
                f"$J^*={J_opt:.3f}$")
        ax_text.text(0.01, 0.38, ostr, fontsize=11,
                     color=OPT_COLOR, fontweight='bold',
                     va='center', transform=ax_text.transAxes)
    elif theta_p is None:
        ax_text.text(0.01, 0.38,
                     'Degenerate constraint  ($c_1=c_2=c_3=0$).',
                     fontsize=11, color='tab:red', va='center',
                     transform=ax_text.transAxes)
    else:
        ax_text.text(0.01, 0.38,
                     'Constrained optimum unavailable (singular KKT).',
                     fontsize=11, color='tab:red', va='center',
                     transform=ax_text.transAxes)

    # Row 4 – hover / selected / hint
    if selected_pt is not None:
        Js = objective_theta(selected_pt)
        sel_str = (f"Selected:   "
                   f"$a_0={selected_pt[0]:+.3f}$,  "
                   f"$a_1={selected_pt[1]:+.3f}$,  "
                   f"$a_2={selected_pt[2]:+.3f}$,  "
                   f"$J={Js:.3f}$")
        ax_text.text(0.01, 0.13, sel_str, fontsize=11, color=SEL_COLOR,
                     va='center', transform=ax_text.transAxes)
    elif hover_pt is not None:
        Jh = objective_theta(hover_pt)
        hov_str = (f"Preview:   "
                   f"$a_0={hover_pt[0]:+.3f}$,  "
                   f"$a_1={hover_pt[1]:+.3f}$,  "
                   f"$a_2={hover_pt[2]:+.3f}$,  "
                   f"$J={Jh:.3f}$")
        ax_text.text(0.01, 0.13, hov_str, fontsize=11, color=HOV_COLOR,
                     va='center', transform=ax_text.transAxes)
    else:
        ax_text.text(0.01, 0.13,
                     'Hover over the constraint plane to preview.  '
                     'Short-click to select a point  (drag still rotates).',
                     fontsize=10, color='#999999', style='italic',
                     va='center', transform=ax_text.transAxes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig = plt.figure(figsize=(15, 8.5))
    fig.canvas.manager.set_window_title(
        'ALADA – Constrained LS Poly: 3-D Parameter Space')

    # ---- axes ----
    ax_scatter = fig.add_axes([0.04, 0.24, 0.33, 0.70])
    ax_3d      = fig.add_axes([0.42, 0.20, 0.56, 0.77], projection='3d')
    ax_text    = fig.add_axes([0.04, 0.00, 0.92, 0.10])
    ax_text.axis('off')

    ax_3d.set_facecolor('#f7f7f7')

    # ---- Generate button ----
    ax_btn = fig.add_axes([0.07, 0.112, 0.10, 0.058])
    btn = Button(ax_btn, 'Generate\nPoints', color='#d0e8f5', hovercolor='#a0c8ef')
    btn.on_clicked(generate_new_points)

    # ---- Reset View button ----
    ax_rvbtn = fig.add_axes([0.22, 0.112, 0.09, 0.058])
    rvbtn = Button(ax_rvbtn, 'Reset\nView', color='#f0e8d5', hovercolor='#e0c89a')
    rvbtn.on_clicked(on_reset_view)

    # ---- Constraint controls ----
    fig.text(0.425, 0.141, 'Constraint:', fontsize=11,
             va='center', ha='left', color='#222222')

    ax_c1 = fig.add_axes([0.510, 0.112, 0.044, 0.050])
    tb_c1 = TextBox(ax_c1, '', initial='1.0', textalignment='center')
    tb_c1.on_submit(on_c1_submit)
    fig.text(0.532, 0.166, '$c_1$', fontsize=12,
             va='bottom', ha='center', color='#555555')
    fig.text(0.558, 0.141, r'$\cdot a_0\;+$', fontsize=11,
             va='center', ha='left', color='#222222')

    ax_c2 = fig.add_axes([0.616, 0.112, 0.044, 0.050])
    tb_c2 = TextBox(ax_c2, '', initial='0.0', textalignment='center')
    tb_c2.on_submit(on_c2_submit)
    fig.text(0.638, 0.166, '$c_2$', fontsize=12,
             va='bottom', ha='center', color='#555555')
    fig.text(0.664, 0.141, r'$\cdot a_1\;+$', fontsize=11,
             va='center', ha='left', color='#222222')

    ax_c3 = fig.add_axes([0.722, 0.112, 0.044, 0.050])
    tb_c3 = TextBox(ax_c3, '', initial='0.0', textalignment='center')
    tb_c3.on_submit(on_c3_submit)
    fig.text(0.744, 0.166, '$c_3$', fontsize=12,
             va='bottom', ha='center', color='#555555')
    fig.text(0.770, 0.141, r'$\cdot a_2\;=$', fontsize=11,
             va='center', ha='left', color='#222222')

    ax_d = fig.add_axes([0.828, 0.112, 0.044, 0.050])
    tb_d = TextBox(ax_d, '', initial='0.0', textalignment='center')
    tb_d.on_submit(on_d_submit)
    fig.text(0.850, 0.166, '$d$', fontsize=12,
             va='bottom', ha='center', color='#555555')

    # ---- events ----
    fig.canvas.mpl_connect('key_press_event',      on_key)
    fig.canvas.mpl_connect('motion_notify_event',  on_hover)
    fig.canvas.mpl_connect('button_press_event',   on_mouse_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)

    generate_new_points()
    plt.show()
