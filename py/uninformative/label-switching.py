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
# ## Conclusion
#
# In contrast to many other methods, nested sampling (NS) is robust to multimodal posteriors given enough live samples. This can be seen in the example (doublet + singlet) as it recovers all $K! = 6$ induced modes (with some imperfections in this particular runs: the mass is not divided equally within all modes). Ironically, this robustness also made it fail for the problem at hand for larger $K$: the profileration of induced modes for any value of $K$ crippled the convergence of our runs.
#
# In short: with any algorithm other than MAP,
# - it's hard to guarantee we've only found the "main" peak (rather than the induced ones) -- if we could do that we could use the $\log K!$ correction, but we can't additional heuristics
# - but it's also impossible to guarantee that we've found all peaks for any nonsmall value of $K$
#
# (In the above it's more correct to say "typical set" rather than "peak".)
#
# So better to use our prior which kills the induced modes by breaking the exchange symmetry.
#
# ## TODO:
# - we can use `dyplot._hist2d()`
# - marginalize out P
# - Layout = 2 rows. First row consists of three adjacent corner plots. Second row is one long plot with log K! on the y axis and the log Z difference on the x axis 
# - log scales?

# +
# %pylab inline
import analyze
import hyper_cmp
import plot

def get_hyperparameters(Q, fs):
    x0 = 200.
    xmax = fs/2
    
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
hyper = get_hyperparameters(Q, fs)

dyplots_kwargs = {'trace_only': False}
# -

jeffreys = analyze.analyze(False, (P,Q), data, hyper, dyplots_kwargs=dyplots_kwargs)

pc = analyze.analyze(True, (P,Q), data, hyper, dyplots_kwargs=dyplots_kwargs)

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
# This is tjhe KEYsss
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

# +
from dynesty import plotting as dyplot

dyplot.traceplot(jeffreys['results'], thin=1, dims=[3,4,5], connect=True, connect_highlight=1, ylim_quantiles=(0,.95));
tight_layout()
# -



res = jeffreys['results'].copy()

res['samples'].shape

res['samples'][:,3:] = sort(res['samples'][:,3:],axis=1)

dyplot.traceplot(res, thin=1, dims=[3,4,5], connect=True, connect_highlight=1, ylim_quantiles=(0,.95));
tight_layout()





pc = analyze.analyze(True, (P,Q), data, hyper, dyplots_kwargs=dyplots_kwargs)

pc['results']['walltime']/jeffreys['results']['walltime']


