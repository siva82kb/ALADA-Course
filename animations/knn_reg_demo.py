"""
Script to demonstrate the k-NN regressor.

Author: Sivakumar Balasubramanian
Date: 19 March 2026
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import platform
if platform.system() == "Windows":
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
else:
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans', 'Liberation Sans']
mpl.rcParams['toolbar'] = 'None'

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# x range for the prediction curve
XPRED = np.linspace(-10, 10, 300)


def knn_regress(x_train, y_train, xnew, k):
    dists = np.abs(x_train - xnew)
    inds = np.argsort(dists)[:k]
    ypred = y_train[inds].mean()
    return ypred, inds


def compute_pred_curve(x_train, y_train, k):
    return np.array([knn_regress(x_train, y_train, xq, k)[0] for xq in XPRED])


def plot_scatter():
    global df, xnew, k, ypred, nn_inds

    # All data points: open circles, tab:blue
    ax.scatter(df['x'], df['y'], edgecolors='tab:blue', facecolors='white',
               s=40, alpha=0.5, zorder=2)

    if xnew is not None:
        # Vertical query line
        ax.axvline(x=xnew, color='black', lw=0.8, zorder=1)

        # Horizontal dotted lines from each neighbor's y to the query line
        for i in nn_inds:
            ax.plot([df['x'].iloc[i], xnew], [df['y'].iloc[i], df['y'].iloc[i]],
                    color='tab:red', linestyle=':', lw=0.8, zorder=3)

        # Highlight k nearest neighbors as filled dots
        ax.scatter(df['x'].iloc[nn_inds], df['y'].iloc[nn_inds],
                   facecolors='tab:blue', edgecolors='tab:blue',
                   s=40, marker='.', zorder=4)

        # Predicted value: red star
        ax.plot(xnew, ypred, color='tab:red', marker='*', markersize=12, zorder=5)

    # Axis styling
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color('#bbbbbb')
    ax.spines['left'].set_color('#bbbbbb')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-10, 10)
    ax.set_ylim(-30, 30)
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)


def plot_pred_curve():
    global pred_curve, xnew, ypred
    axins.cla()

    # Faded scatter
    axins.scatter(df['x'], df['y'], edgecolors='tab:blue', facecolors='white',
                  s=20, alpha=0.3, zorder=1)

    # Prediction curve
    axins.plot(XPRED, pred_curve, color='tab:red', lw=1.5, zorder=2)

    # Current query point on the curve
    if xnew is not None:
        axins.plot(xnew, ypred, color='tab:red', marker='*', markersize=8, zorder=3)

    axins.set_xlim(-10, 10)
    axins.set_ylim(-30, 30)
    axins.spines['right'].set_visible(False)
    axins.spines['top'].set_visible(False)
    axins.spines['bottom'].set_color('#bbbbbb')
    axins.spines['left'].set_color('#bbbbbb')
    axins.tick_params(axis='both', colors='#bbbbbb', labelsize=8)
    axins.set_xlabel(f'{k}-NN prediction curve', fontsize=10, color='#888888')


def update_text():
    global k, xnew, ypred
    ax2.cla()
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-1.1, 1.2)

    xpos, ypos, delypos = 0.1, 0.75, 0.12

    # Instructions
    ax2.text(0.1, 1.20,
             'Click on the plot to place a query point.',
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')
    ax2.text(0.1, 1.12,
             r'Use 1-9 to select $k$.',
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')

    j = 0
    ax2.text(xpos, ypos - j * delypos, f"k = {k:2d}", fontsize=14)
    j += 1

    if xnew is not None:
        ax2.text(xpos, ypos - j * delypos,
                 f"$x_{{new}}$ = {xnew:.2f}",
                 fontsize=14)
        j += 1
        ax2.text(xpos, ypos - j * delypos,
                 f"$\\hat{{y}}$ = {ypred:.2f}",
                 fontsize=14, color='tab:red')


def redraw():
    ax.cla()
    update_text()
    plot_scatter()
    plot_pred_curve()
    fig.canvas.draw()


def on_click(event):
    global xnew, ypred, nn_inds
    if event.inaxes != ax:
        return
    xnew = event.xdata
    ypred, nn_inds = knn_regress(df['x'].values, df['y'].values, xnew, k)
    redraw()


def on_press(event):
    global k, xnew, ypred, nn_inds, pred_curve

    if event.key == 'escape':
        plt.close(fig)
        return

    if event.key in [f'{i}' for i in range(1, 10)]:
        k = int(event.key)
        pred_curve = compute_pred_curve(df['x'].values, df['y'].values, k)
        if xnew is not None:
            ypred, nn_inds = knn_regress(df['x'].values, df['y'].values, xnew, k)
        redraw()


if __name__ == "__main__":
    # Generate regression data (matches notebook)
    np.random.seed(2)
    n_samples = 100
    x = np.random.randn(n_samples) * 4
    y = 0.1 * x ** 3 - 0.5 * x ** 2 - 2 * x + 3 + np.random.randn(n_samples) * 6
    df = pd.DataFrame({'x': x, 'y': y})

    # State variables
    k = 5
    xnew = None
    ypred = None
    nn_inds = None
    pred_curve = compute_pred_curve(df['x'].values, df['y'].values, k)

    # Figure layout (matches kmeans_demo / knn_demo)
    fig = plt.figure(figsize=(13, 7.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=(1.2, 0.1, 1))
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    axins = inset_axes(ax2, width="90%", height="50%", loc=4, borderpad=1)
    axins.axis('off')
    fig.canvas.manager.set_window_title('ALADA k-NN Regression Animation')

    # Initial plot
    redraw()

    # Connect events
    fig.canvas.mpl_connect('key_press_event', on_press)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout(pad=3)
    plt.show()