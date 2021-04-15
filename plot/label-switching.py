# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Illustration of label-switching
#
# Case at hand:
#
# ```
# # ə in /ənˈtɪl/ ("until")
# # F0 = 110 Hz
# ```
#
# Doublet (R1 and R2) and separated resonance (R3)
#
# ```
# ╒═════════╤════════════╤═════════════╕
# │ R1      │ R2         │ R3          │
# ╞═════════╪════════════╪═════════════╡
# │ 147(31) │ 71(14)     │ 70.5(4.9)   │
# ├─────────┼────────────┼─────────────┤
# │ 511(21) │ 605.6(8.0) │ 1439.2(2.4) │
# ╘═════════╧════════════╧═════════════╛
# ```
#
# Even though the frequency difference may look large, this is still a doublet for us because the wide glottal pulse smears out the spectrum such that the two resonances are not resolved in the spectrum.
#
# The scale of doublets (i.e. "what is close to each other"/"what can we resolve in the spectrum") is thus determined by the width of the glottal pulse.
#
# Why not marginalize out $P$? Because results are less clear, so stick with $P=7$ (is the MAP for both priors with a posterior probability of about 75%).
#
# ## Conclusion
#
# In contrast to many other methods, nested sampling (NS) is robust to multimodal posteriors given enough live samples. This can be seen in the example (doublet + singlet) as it recovers all $K! = 6$ induced modes (with some imperfections in this particular runs: the mass is not divided equally within all modes). Ironically, this robustness also made it fail for the problem at hand for larger $K$: the profileration of induced modes for any value of $K$ crippled the convergence of our runs.
#
# In short: with any algorithm other than MAP,
# - it's hard to guarantee we've only found the "main" peak (rather than the induced ones) -- if we could do that we could use the $\log K!$ correction, but we can't without additional heuristics
# - but it's also impossible to guarantee that we've found all peaks for any nonsmall value of $K$
#
# (In the above it's more correct to say "typical set" rather than "peak".)
#
# So better to use our prior which kills the induced modes by breaking the exchange symmetry.

# %run plot.ipy
# %matplotlib inline

# +
import analyze
import hyper_cmp
import plot

def get_hyperparameters(Q, x0, xmax):    
    bounds = {
        'b': [(10.,)*Q, (500.,)*Q],
        'f': [(x0,)*Q, (xmax,)*Q]
    }

    Ex = np.array([500., 1000., 1500., 2000., 2500.]) # Schwa model
    F = [x0, *Ex]
    
    delta = 1.
    
    return bounds, F, delta

P = 7
Q = 3
data = hyper_cmp.get_data('slt/arctic_b0041')
fs = data[0]

x0 = 200.
xmax = fs/2

hyper = get_hyperparameters(Q, x0, xmax)

dyplots_kwargs = {'trace_only': False}
# -

jeffreys = analyze.analyze(False, (P,Q), data, hyper, dyplots_kwargs=dyplots_kwargs)
#jeffreys = analyze.analyze_average(False, Q, data, hyper)

pc = analyze.analyze(True, (P,Q), data, hyper, dyplots_kwargs=dyplots_kwargs)
#pc = analyze.analyze_average(True, Q, data, hyper)

# +
#https://stackoverflow.com/a/21918893/6783015
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = jeffreys['results']['samples'][:,3:].T
w = analyze.exp_and_normalize(jeffreys['results']['logwt'])

keep = w > 1e-5
x = x[:,keep]
w = w[keep]


mu=np.array([1,10,20])
sigma=np.matrix([[4,10,0],[10,25,0],[0,0,100]])
data=np.random.multivariate_normal(mu,sigma,1000)
values = data.T

kde = stats.gaussian_kde(x, weights=w)
density = kde(x)

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
#x, y, z = values
ax.scatter(*x, c=density)
plt.show()

# All 3! = 6 modes are found but mode death occured for the two top left in the picture below
# So the mass in the modes is not approximately equal

# +
from dynesty import plotting as dyplot

fig, subplots = dyplot.cornerplot(
    jeffreys['results'],
    dims=[3,4,5],
    span=([300, 1600],)*3,
    quantiles_2d=[0.4, 0.85, 0.99], # Replace with 2D credible intervals at 1, 2, 3 sigma
    color='black'
)

dyplot.cornerplot(
    pc['results'],
    dims=[3,4,5],
    span=([300, 1600],)*3,
    quantiles_2d=[0.4, 0.85, 0.99], # Replace with 2D credible intervals at 1, 2, 3 sigma
    fig=(fig,subplots),
    color='blue'
)
# -

pc['results']['walltime']/jeffreys['results']['walltime'] # Also about 10% speedup in this particular case

# +
import matplotlib.ticker as mticker

limits = array([x0,xmax])
bisect = logspace(*log10(limits),3)


sigmas = array([1, 2, 3])
levels = 1.0 - np.exp(-0.5 * sigmas ** 2)

def plot_pair(i, j, ax=None, show_legend=False, yticklabels=False):
    colors = [COLOR_JEFFREYS2, COLOR_PC]
    for a, c, impose in zip([jeffreys, pc], colors, (False, True)):
        x = a['results']['samples'][:,3:].T
        w = analyze.exp_and_normalize(a['results']['logwt'])

        ax = plot.plot_frequency_cornerplot(
            x[i-1], x[j-1],
            span=[limits,limits],
            weights=w,
            ax=ax,
            smooth = 0.0175,
            levels = levels,
            plot_contours=True,
            fill_contours=True,
            plot_density=False,
            plot_datapoints=False,
            color=c,
            impose_ordering=impose
        )

    ax.fill_between(bisect, bisect, y2=limits[1], color='grey', alpha=ORDERED_ZONE_ALPHA) # Ordered zone
    ax.plot(bisect, bisect, ':', lw=1, color = 'black')
    
    xy = array([.7, .75])
    ax.text(*(xy*1.03), f"$x_{j} > x_{i}$", transform=ax.transAxes, rotation=45, rotation_mode='anchor')

    ax.set_aspect('equal', 'box')
    ax.set(xlim=(x0, xmax), ylim=(x0, xmax))
    
    ax.set_xlabel(f"$x_{i}$ [Hz]")
    ax.set_ylabel(f"$x_{j}$ [Hz]")
    
    ticks = [x0, 500, 1000, xmax]
    labels = list(map(int, ticks))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels if yticklabels else [])
    
    if show_legend:
        proxy = [plt.Rectangle((0,0),1,1,fc = c) for c in colors]
        ax.legend(proxy, [r'$P_1(x_k,x_\ell)$', r'$P_3(x_k,x_\ell)$'], loc = "upper left")

fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.5), constrained_layout=True)

plot_pair(1, 2, axes[0], True, True)
plot_pair(1, 3, axes[1])
plot_pair(2, 3, axes[2])
save_plot(fig, 'label-switching.pgf')
# -

