"""
Script to demonstrate the k-NN classifier.

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


# Class colors (matching notebook style)
clustcolors = [
    'tab:blue',
    'tab:red',
    'tab:green',
    'tab:orange',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan'
]

N_CLASSES = 3


def euclidean_distances(Xvals, xnew):
    return np.sqrt(np.sum((Xvals - xnew) ** 2, axis=1))


def knn_predict(Xvals, yvals, xnew, k):
    dists = euclidean_distances(Xvals, xnew)
    inds = np.argsort(dists)[:k]
    votes = yvals[inds].astype(int)
    counts = np.bincount(votes, minlength=N_CLASSES)
    pred = int(np.argmax(counts))
    return pred, inds, counts


def generate_cluster(n_samples, n_features, cluster_std, center):
    return np.random.randn(n_samples, n_features) * cluster_std + center


def generate_clusters(n_clusters, n_samples, n_features):
    np.random.seed(2)
    cluster_std = np.array([0.8, 1.0, 1.2])
    centers = np.array([[2, 2], [-2, -1], [2.5, -2.5]])
    X = np.array([
        generate_cluster(n_samples, n_features, cluster_std[i], centers[i])
        for i in range(n_clusters)
    ])
    nc = len(X)
    ns = len(X[0])
    Xorg = np.zeros((nc * ns, 3))
    Xorg[:ns, 0:2] = X[0]
    Xorg[:ns, 2] = 0
    Xorg[ns:2*ns, 0:2] = X[1]
    Xorg[ns:2*ns, 2] = 1
    Xorg[2*ns:3*ns, 0:2] = X[2]
    Xorg[2*ns:3*ns, 2] = 2
    return pd.DataFrame(Xorg, columns=['x1', 'x2', 'cluster'])


def plot_scatter():
    global X, xnew, k, pred_class, nn_inds

    # All training points: open circles colored by class
    for c in range(N_CLASSES):
        mask = X['cluster'].values == c
        ax.scatter(X['x1'].values[mask], X['x2'].values[mask],
                   edgecolors=clustcolors[c], facecolors='white',
                   s=50, alpha=0.75, zorder=2)

    if xnew is not None:
        # Lines from query point to each of the k nearest neighbors
        for i in nn_inds:
            c = int(X['cluster'].values[i])
            ax.plot([X['x1'].values[i], xnew[0]],
                    [X['x2'].values[i], xnew[1]],
                    color=clustcolors[c], lw=1, zorder=1, alpha=0.8)

        # Highlight the k nearest neighbors as filled circles
        for i in nn_inds:
            c = int(X['cluster'].values[i])
            ax.scatter(X['x1'].values[i], X['x2'].values[i],
                       edgecolors=clustcolors[c], facecolors=clustcolors[c],
                       s=50, zorder=3)

        # Query point: filled square colored by predicted class
        ax.scatter(xnew[0], xnew[1],
                   edgecolors='black', facecolors=clustcolors[pred_class],
                   s=100, marker='s', zorder=4)

    # Axis styling (matches kmeans_demo)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('#bbbbbb')
    ax.spines['left'].set_color('#bbbbbb')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)


def plot_vote_chart():
    global vote_counts, pred_class
    axins.cla()
    if vote_counts is None:
        axins.axis('off')
        return

    axins.bar(range(N_CLASSES), vote_counts,
              color=[clustcolors[c] for c in range(N_CLASSES)],
              alpha=0.7, width=0.5)
    axins.set_xticks(range(N_CLASSES))
    axins.set_xticklabels([f'C{c + 1}' for c in range(N_CLASSES)],
                           fontsize=12, color='#bbbbbb')
    axins.set_ylim(0, k + 0.5)
    axins.set_yticks(range(k + 1))
    axins.spines['right'].set_visible(False)
    axins.spines['top'].set_visible(False)
    axins.spines['bottom'].set_color('#bbbbbb')
    axins.spines['left'].set_color('#bbbbbb')
    axins.tick_params(axis='both', colors='#bbbbbb')
    axins.set_xlabel('Class', fontsize=12)
    axins.set_ylabel('Votes', fontsize=12)


def update_text():
    global k, pred_class, vote_counts
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

    if pred_class is not None:
        ax2.text(xpos, ypos - j * delypos,
                 f"Predicted class: {pred_class + 1}",
                 fontsize=14, color=clustcolors[pred_class])
        j += 1
        vote_str = '   '.join(
            [f'C{c + 1}: {vote_counts[c]}' for c in range(N_CLASSES)]
        )
        ax2.text(xpos, ypos - j * delypos,
                 f"Votes — {vote_str}",
                 fontsize=12)


def on_click(event):
    global xnew, pred_class, nn_inds, vote_counts
    if event.inaxes != ax:
        return
    xnew = np.array([event.xdata, event.ydata])
    pred_class, nn_inds, vote_counts = knn_predict(
        X[['x1', 'x2']].values, X['cluster'].values, xnew, k
    )
    ax.cla()
    update_text()
    plot_scatter()
    plot_vote_chart()
    fig.canvas.draw()


def on_press(event):
    global k, xnew, pred_class, nn_inds, vote_counts

    if event.key == 'escape':
        plt.close(fig)
        return

    if event.key in [f'{i}' for i in range(1, 10)]:
        k = int(event.key)
        if xnew is not None:
            pred_class, nn_inds, vote_counts = knn_predict(
                X[['x1', 'x2']].values, X['cluster'].values, xnew, k
            )
        ax.cla()
        update_text()
        plot_scatter()
        plot_vote_chart()
        fig.canvas.draw()


if __name__ == "__main__":
    # Generate training data (same as kmeans_demo)
    X = generate_clusters(n_clusters=3, n_samples=25, n_features=2)

    # State variables
    k = 3
    xnew = None
    pred_class = None
    nn_inds = None
    vote_counts = None

    # Figure layout (matches kmeans_demo)
    fig = plt.figure(figsize=(13, 7.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=(1.2, 0.1, 1))
    ax = fig.add_subplot(gs[0, 0])
    ax.equal_aspect = True
    ax.axis('equal')
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    axins = inset_axes(ax2, width="90%", height="50%", loc=4, borderpad=1)
    axins.axis('off')
    fig.canvas.manager.set_window_title('ALADA k-NN Animation')

    # Initial plot
    ax.cla()
    update_text()
    plot_scatter()
    plot_vote_chart()
    fig.canvas.draw()

    # Connect events
    fig.canvas.mpl_connect('key_press_event', on_press)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout(pad=3)
    plt.show()