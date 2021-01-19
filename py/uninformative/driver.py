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
# - The new prior helped uncover a fifth frequency. F1 has been resolved into two frequencies with overwhelming evidence. The reason we did not see this is because of non-overlapping intervals with the old prior, which precluded F2 getting near F1.
#
# - Original code contained serious errors which, strangely, did not affect the results that much. See `errors.md`.
#
# - The prior bounds for $\alpha$ and $\omega$ were chosen given Praat's estimates, so we choose the hyperparameters given that we know that it was `/ae/`. From Vallee (1994) we get: $\langle \boldsymbol x \rangle = (610, 1706, 2450, 3795)$. We then round these numbers. The lowerbound $x_0$ is chosen as the lower bound of $F_1$'s interval used in the paper.
#
# - Note that the prior bounds for $F$ have very little overlap: kind of like cheating.
#
# - Current thoughts: new priors are indeed *very* uninformative (also have less bounds to set), so higher $H$ and longer runtimes compared to prior with very definite, hardly overlapping bounds.
#
# - After running, find the "true value" of the bandwidths $B$. Then set these to fixed values so we can compare differences in the prior of frequencies only.
#
# - Try wider bounds with actual overlap to see what the effect is.
#
# - We could compare our prior to a range of nested sampling runs with lognormals as priors with different relative uncertainty $\rho$ and then see to roughly which value of $\rho$ our prior corresponds by comparing the information (i.e. similar information as our prior for $\rho = \rho^*$ $\rightarrow$ our prior is like $\rho^*\%$ uncertainty).
#
# - We can show some transfer functions with error margins (GP style) as a condensed way to visualize the estimated frequencies.
#
# ## Nested sampling options
#
# The type of bounding and sampling methods used are *crucial*.
#
# The likelihood function with the new prior exhibits several sharp peaks within each frequency.  
# Nested sampling performance is improved (and indeed actually finishes runs) by:  
#
# - Using 'multi' as the bound method -- is the only viable alternative  
# - Using random walk 'rslice' instead of uniform sampling. The order of preference of sampling  
#   methods is: 'unif' < 'slice' ~ 'hslice' < 'rstagger' < 'rwalk' < 'rslice'
#   
# - Note that `rslice` is similar to Galilean Monte Carlo [@Speagle2019]
#
# - Using bootstrapping to estimate expansion factor instead of the fixed default.  
#        
#   > Bootstrapping these expansion factors can help to ensure accurate evidence estimation    
#   > when the proposals rely heavily on the size of an object rather than the overall shape,  
#   > such as when proposing new points uniformly within their boundaries. In theory, it also  
#   > helps to prevent mode "death": if occasionally a secondary mode disappears when bootstrapping,  
#   > the existing bounds would be expanded to theoretically encompass it. In practice, however,
#   > most modes are widely separated, [as in our case] leading enormous expansion factors  
#   > whenever any possible instance of mode death may occur.[1]  
#  
# [1]: https://github.com/joshspeagle/dynesty/blob/master/demos/Examples%20--%20Gaussian%20Shells.ipynb 

# +
# %pylab inline

import joblib
import model
import analyze

# +
# %run driver.ipy

PQ_grid = get_PQ_grid(5, 10)
data = get_data('bdl/arctic_a0017', 11000)
hyper = get_hyperparameters()

# +
# The all-important parameters for nested sampling
samplerargs = {'nlive': 500, 'bound': 'multi', 'sample': 'rslice', 'bootstrap': 10}
runargs = {'save_bounds': False}

# Output to stdout is not printed in the notebook. If this is desired,
# change backend to "multiprocessing" and uncomment the decorator
# @memory.cache of model.run_nested()
options = dict(n_jobs=6, verbose=50, timeout=60*60*2, backend=None)

runid = 0

with joblib.Parallel(**options) as parallel:
    parallel(
        joblib.delayed(model.run_nested)(
            new, order, data, hyper, runid=runid
        ) for order in PQ_grid for new in (False, True)
    )
