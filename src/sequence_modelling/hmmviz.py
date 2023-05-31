# -*- coding: utf-8 -*-
"""
Visualization utils

@author: nbhushan

"""

import numpy as np

# import pdb
import sequence_modelling.DiscreteOptim as optim

# from matplotlib import cm as CM


def view_viterbi(ax, obs, paths, mean, seq):
    """Visualize the Viterbi sequence."""
    # pdb.set_trace()
    path = paths[seq]
    y = obs[seq].flatten()
    z = path
    m = mean.flatten()
    ordr = np.argsort(m)
    z = np.argsort(ordr)[z]
    m = m[ordr]
    # pdb.set_trace()
    r = np.arange(len(y))
    ax.plot(r, y)
    for k, v in enumerate(m):
        t = z == k
        ax.plot(
            r[t],
            v * np.ones((t.sum(),), float),
            ".",
            label="State {0}: {1:g}".format(k, v),
        )
    ax.set_title("Viterbi State sequence; seq : " + str(seq))
    ax.set_xlabel("time")
    ax.set_ylabel("obs")
    ax.legend(prop=dict(size="xx-small"))


def view_postduration(ax, obs, path, mean, reslist, ranknlist, seq):
    """Visualize the estimated state durations based on the posterior
    distribution.
    """
    path = path[seq]
    res = reslist[seq]
    rankn = ranknlist[seq]
    clr = ["m", "c", "r", "g"]
    y = obs[seq].flatten()
    z = path
    m = mean.flatten()
    ordr = np.argsort(m)
    z = np.argsort(ordr)[z]
    m = m[ordr]
    r = np.arange(len(y))
    ax.plot(r, y)
    for k, v in enumerate(m):
        t = z == k
        ax.plot(
            r[t],
            v * np.ones((t.sum(),), float),
            ".",
            c=clr[::-1][k],
            label="State {0}: {1:g}".format(k, v),
        )
    ax.set_title("HMM State duration estimation, seq :" + str(seq))
    ax.set_xlabel("time")
    ax.set_ylabel("obs")
    ax.legend(prop=dict(size="xx-small"))
    for k in range(1, len(m) - 1):
        c = np.argsort(rankn[k])
        for label, x, z in zip(
            np.char.mod("%.2f", res[k][c]),
            rankn[k][c] + 1,
            [mean.flatten()[k]] * res.shape[1],
        ):
            ax.annotate(
                label,
                xy=(x, z),
                xytext=(-20, 20),
                textcoords="offset points",
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", fc=clr[k], alpha=0.4),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )
    ax.legend(prop=dict(size="xx-large"))


def view_EMconvergence(ay, ll):
    """Visualize the variation of the llh during each iteration of
    the EM algorithm. Note: THE LLH MUST MONOTONICALLY INCREASE.
    """
    x = list(range(len(ll)))
    ay.plot(x, ll)
    ay.set_title("EM convergence")
    ay.set_ylabel("Log-likelihood")
    ay.set_xlabel("Iteration")


def view_ksi(obs, ksi):
    from matplotlib.pyplot import figure, show

    fa = figure()
    obs = obs[0][0]
    ksi = np.exp(ksi)
    az = fa.add_subplot(1, 1, 1)
    az.plot(obs, label="Power Data (W)")
    x = np.arange(len(obs))
    y = obs
    z = ksi[:, 0, 1]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    az.pcolor(x[:-1], x[:-1], z, cmap="RdBu", vmin=z_min, vmax=z_max)
    # az.pcolor(x[:-1], y[:-1], z, cmap='RdBu', vmin=z_min, vmax=z_max)
    az.colorbar()
    # az.plot(ksi[:,0, 1]*500, label = "Active to Idle")
    # az.plot(ksi[:,1, 2]*100, label = "Idle to Sleep")
    az.set_ylabel("obs")
    az.set_title("Posterior probability of transitions from Active to Idle ")
    az.legend()
    show()


def view_statedurations(fc, path, K):
    """View histograms of state duration.

    The frequencies are obtained from the Viterbi sequence.
    """
    from itertools import groupby

    lengths = [None] * K
    for k in range(K):
        a = path == k
        lengths[k] = [sum(g) for b, g in groupby(a) if b]
    aw = fc.add_subplot(2, 2, 1)
    ax = fc.add_subplot(2, 2, 2)
    ay = fc.add_subplot(2, 2, 3)
    az = fc.add_subplot(2, 2, 4)
    fc.suptitle("Histogram of state duration distributions")
    aw.set_xlabel("Duration (s)")
    aw.set_ylabel("Probability")
    # labels = ['active', 'idle', 'sleep', 'deep sleep']
    n, binsw, patches = aw.hist(
        lengths[0], normed=1, histtype="stepfilled", label="active"
    )
    n, binsx, patches = ax.hist(
        lengths[1], normed=1, histtype="stepfilled", label="idle"
    )
    n, binsy, patches = ay.hist(
        lengths[2], normed=1, histtype="stepfilled", label="sleep"
    )
    n, binsz, patches = az.hist(
        lengths[3], normed=1, histtype="stepfilled", label="deep sleep"
    )
    aw.legend()
    ax.legend()
    ay.legend()
    az.legend()


def plotcontour(K, taus, weights, obs, history, title):
    """Visualize the combinatorial search.

    Visualize the propogation of the search using contour lines.
    """
    # pdb.set_trace()
    step = np.ceil(np.min(np.diff(taus)) / 3)
    tau1 = taus[1]
    tau2 = taus[2]

    x = np.arange(tau1 - step, tau1 + step, 1)
    y = np.arange(tau2 - step, tau2 + step, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for a, i in enumerate(x):
        for b, j in enumerate(y):
            Z[b, a] = optim.objective(K, [np.array([0, i, j])], obs, weights)[0]
    from matplotlib.pyplot import figure

    fa = figure()
    ax = fa.add_subplot(111)
    CS = ax.contour(X, Y, Z)
    # labels=['genetic algorithm', 'local search', 'simulated annealing']
    x = []
    y = []
    # style=['y-o', 'r-o', 'g-o']
    for taus in history:
        x.append(taus[1])
        y.append(taus[2])
    ax.plot(x[0], y[0], "r-o", label="initial configuration")
    ax.plot(x[1:-1], y[1:-1], "o")
    ax.plot(x[-1], y[-1], "g-o", label="final configuration")
    # Label contours
    ax.clabel(CS, inline=1, fontsize=10)
    # Add some text to the plot
    ax.set_title(title)
    ax.set_xlabel("tau1")
    ax.set_ylabel("tau2")
    ax.legend(prop=dict(size="xx-small"))
    fa.savefig("C:\\Local\\FINALEXPERIMENTS\\4287\\" + "4287" + title + ".png")
