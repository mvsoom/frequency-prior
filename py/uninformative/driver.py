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
# - The new prior helped uncover a fifth frequency. F1 has been resolved into two frequencies with overwhelming evidence. The reason we did not see this is because of non-overlapping intervals with the old prior, which precluded F2 getting near F1. We could say that the new prior allows to pick up all the vocal tract *resonancies* which are more fine grained than the canonical formants we are used to; in this example we were able to resolve F1 "doublets".
#
# - Original code contained serious errors which, strangely, did not affect the results that much. See `errors.md`.
#
# - After running, find the "true value" of the bandwidths $B$. Then set these to fixed values and rerun analysis so we can compare differences in the prior of frequencies only.
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
import analyze
import driver_script
import aux

# Allows you to see progress (garbled)
# #%run driver_script.py "bdl/arctic_a0017"

# Execute this line to suppress dynesty's output to stderr
# # %python3 driver_script.py "bdl/arctic_a0017" 2>/dev/null

# +
data = aux.get_data("bdl/arctic_a0017", 11000)
hyper = aux.get_hyperparameters()

analyze.analyze(driver_script.run_nested(False, 0, 2, data, hyper))
# -



