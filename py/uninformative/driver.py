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

# # Driver script
#
# - Original code contained serious errors. See `errors.md`.
#
# - The prior bounds for $\alpha$ and $\omega$ were chosen given Praat's estimates, so we choose the hyperparameters given that we know that it was `/ae/`. From Vallee (1994) we get: $\langle \boldsymbol x \rangle = (610, 1706, 2450, 3795)$. We then round these numbers. The lowerbound $x_0$ is chosen as the lower bound of $F_1$'s interval used in the paper.
#
# - Note that the prior bounds for $F$ have very little overlap: kind of like cheating.
#
# - Current thoughts: new priors are indeed *very* uninformative (also have less bounds to set), so higher $H$ and longer runtimes compared to prior with very definite, hardly overlapping bounds.

# +
# %pylab inline

import joblib
import model
import analyze

# +
# %run driver.ipy

PQ_grid = get_PQ_grid()
data = get_data()
hyper = get_hyperparameters()

# +
# Output to stdout is not printed in the notebook. If this is desired,
# change backend to "multiprocessing" and uncomment the decorator
# @memory.cache of model.run_nested()
options = dict(n_jobs=7, verbose=50, timeout=None, backend=None)

runid = 0

with joblib.Parallel(**options) as parallel:
    parallel(
        joblib.delayed(model.run_nested)(
            new, order, data, hyper, runid=runid
        ) for order in PQ_grid for new in (False, True)
    )
# -


