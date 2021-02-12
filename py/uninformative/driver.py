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
# ## Current session
#
# The PC is running `driver_pc.sh`, the Mac is running `driver_mac.sh`.
#
# **Update:** Finished overnight and caches have been transferred using `transfer_cache_*.sh` scripts. But joblib doesn't seem to use the caches.
#
# ## Notes
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
#data = aux.get_data("bdl/arctic_a0017", 11000)
data = aux.get_data("rms/arctic_a0382", 11000)
hyper = aux.get_hyperparameters()

analyze.analyze(driver_script.run_nested(False, 10, 5, data, hyper))


# +
import pandas
d = []

for new, P, Q in aux.get_grid(10, 5):
    res = driver_script.run_nested(new, P, Q, data, hyper)
    print(new, P, Q, res.logz[-1], sep='\t')
    d.append([new, P, Q, res.logz[-1], res.walltime])

d = pandas.DataFrame(d, columns=('new', 'P', 'Q', 'lz', 'walltime'))
# -

d.sort_values('lz')

analyze.analyze(driver_script.run_nested(True, 10, 5, data, hyper))

analyze.analyze(driver_script.run_nested(False, 5, 4, data, hyper))

res

r.information[-1],

# +
output_file = "post/run_stats.csv"

help(csv.DictWriter)
# -

a = diff(r.logl) > 0
a[-1000:]

# + active=""
#
