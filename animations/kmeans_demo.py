"""
Script to demonstrate The k-means algorithm.

Author: Sivakumar Balasubramanian
Date: 21 August 2024
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import platform
if platform.system() == "Windows":
    mpl.rc('font',**{'family':'Times New Roman', 'sans-serif': 'Arial'})
else:
    mpl.rc('font',**{'family':'Helvetica', 'sans-serif': 'Helvetica'})
mpl.rcParams['toolbar'] = 'None' 

import sys
sys.path.append("../")
from aladalib import chap01 as ch01


clustcolors = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan'
]

def reset_params():
    global X, km, iter, cm, ca, J
    # Reset the solution
    km = ch01.KMeans(X=X.values, k=k)
    iter = 0
    ca, cm, J = cm, ca, J = km.fit(max_iter=25, cost_change_th=0.0)


def update():
    global funclass, ecfunclass, t, xc, norms, tangs
    pass


def plot_scatter():
    global X, funclass, ecfunclass, t, xc, norms, tangs
    if iter < 0:
        ax.plot(X['x1'], X['x2'], color='black', marker='o', linestyle='',
                markerfacecolor='white', markersize=7, alpha=1.0)
    else:
        minx = iter // 2
        ainx = (iter // 2) - 1 if iter % 2 == 0  else iter // 2
        if ainx == -1:
            ax.plot(X['x1'], X['x2'], color='black', marker='o', linestyle='',
                    markerfacecolor='white', markersize=7, alpha=1.0, zorder=1)
        for _k in range(k):
            # Plot the cluster means
            ax.plot(cm[minx, _k, 0], cm[minx, _k, 1],
                    marker='s', color=clustcolors[_k],
                    markerfacecolor=clustcolors[_k],
                    markersize=10, zorder=3)
            # Plot the cluster mean trajectory so far.
            ax.plot(cm[:minx+1, _k, 0], cm[:minx+1, _k, 1],
                    lw=1, color=clustcolors[_k], zorder=1)
            # Plot the cluster assignments.
            if ainx == -1:
                continue
            ax.plot(X['x1'].values[ca[ainx, :] == _k],
                    X['x2'].values[ca[ainx, :] == _k], 
                    color=clustcolors[_k], marker='o', linestyle='',
                    markerfacecolor='white', markersize=7, alpha=1.0,
                    zorder=2)
    
    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('#bbbbbb')
    ax.spines['left'].set_color('#bbbbbb')

    # Set xlims and ylims
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def plot_const_func():
    # Update the inset plot.
    if iter % 2 == 0:
        axins.cla()
        axins.plot(range(iter // 2), J[:iter //2], color="black",
                   lw=1, marker='o', markersize=4, markerfacecolor='white')
        # Axis limits
        axins.set_xlim(-1, len(J))
        axins.set_ylim(0.8 * np.min(J), 1.2 * np.max(J))
        axins.spines["right"].set_visible(False)
        axins.spines["top"].set_visible(False)
        axins.spines["left"].set_position(("axes", -0.05))
        axins.spines['bottom'].set_color('#bbbbbb')
        axins.spines['left'].set_color('#bbbbbb')
        axins.tick_params(axis='both', colors='#bbbbbb')
        axins.set_xlabel("iter", fontsize = 14)


def update_text():
    global k
    # Remove the previous text
    ax2.cla()
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-1.1, 1.2)

    # Text positions
    xpos, ypos, delypos = 0.1, 0.75, 0.1

    # Instruction.
    ax2.text(0.1, 1.2,
             r'Press enter to restart with a new set of random cluster means.',
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')
    ax2.text(0.1, 1.12,
             r'Use 1-9 to select $k$.',
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')
    ax2.text(0.1, 1.04,
             r'Use the right/left arrows to increment or decrement iteration.',
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')

    # Current point detials
    if iter < 0:
        return
    j = 0
    ax2.text(xpos, ypos - j * delypos,
             f"k = {k:2d} \t Iteration = {iter:2d}",
             fontsize=14)
    j += 1
    ax2.text(xpos, ypos - j * delypos,
             f"$J_{{clust}}$ = {J[iter // 2]:8.2f}",
             fontsize=14)


# Handling key press events
def on_press(event):
    global k, iter, km, cm, ca, J
    print(event.key)
    # Close figure if escaped.
    if event.key == 'escape':
        plt.close(fig)
        return

    # Chekc if the solution needs to be updated.
    if event.key == 'right' and iter >= 0:
        # Increment iter
        iter = iter + 1 if (iter + 1) < len(J) else iter
        ax.cla()
    elif event.key == 'left' and iter >= 0:
        # Decrement iter
        iter = iter - 1 if (iter - 1) >= 0 else iter
        ax.cla()
    elif event.key in [f'{i}' for i in range(1, 10)]:
        # Reset the plot
        k = int(event.key)
        reset_params()
        ax.cla()
    elif event.key == 'enter':
        # Refit k-means
        iter = 0
        cm, ca, J = km.fit(max_iter=25, cost_change_th=0.0)
        ax.cla()
    
    # Draw the plot and text
    ax.cla()
    update_text()
    plot_scatter()
    plot_const_func()
    fig.canvas.draw()


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
    # Create a pandas dataframe
    nc = len(X)
    ns = len(X[0])
    Xorg = np.zeros((nc * ns, 3))
    Xorg[:ns, 0:2] = X[0]
    Xorg[:ns, 2] = 0
    Xorg[ns:2*ns, 0:2] = X[1]
    Xorg[ns:2*ns, 2] = 1
    Xorg[2*ns:3*ns, 0:2] = X[2]
    Xorg[2*ns:3*ns, 2] = 2

    # Form a dataframe
    return pd.DataFrame(Xorg, columns=['x1', 'x2', 'cluster'])

if __name__ == "__main__":
    # Generate random scatter plot.
    X = generate_clusters(n_clusters = 3, n_samples = 25, n_features = 2)
    
    # Variables to be used.
    k = 3
    km = None
    iter = -1
    km = None
    cm, ca, J = None, None, None
    reset_params()
    
    # Create the figure and the axis.
    fig = plt.figure(figsize=(13, 7.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=(1.2, 0.1, 1))
    ax = fig.add_subplot(gs[0, 0])
    ax.equal_aspect = True
    ax.axis('equal')
    # ax.axis('off')
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    axins = inset_axes(ax2, width="90%", height="50%", loc=4, borderpad=1)
    axins.axis('off')
    fig.canvas.manager.set_window_title('ALADA k-Means Animation')

    # Plot stuff
    ax.cla()
    update_text()
    plot_scatter()
    plot_const_func()
    fig.canvas.draw()

    # Create the figure and the axis.
    fig.canvas.mpl_connect('key_press_event', on_press)
    plt.tight_layout(pad=3)
    plt.show()