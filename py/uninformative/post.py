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

# # Post analysis of the nested sampling runs
#
# Cases are ordered in terms of niceness; the nices examples are first.
#
# ## Notes
#
# - The new prior helped uncover a fifth frequency. F1 has been resolved into two frequencies with overwhelming evidence. The reason we did not see this is because of non-overlapping intervals with the old prior, which precluded F2 getting near F1. We could say that the new prior allows to pick up all the vocal tract *resonancies* which are more fine grained than the canonical formants we are used to; in this example we were able to resolve F1 "doublets".
#
# - After running, find the "true value" of the bandwidths $B$. Then set these to fixed values and rerun analysis so we can compare differences in the prior of frequencies only.
#
# - We could compare our prior to a range of nested sampling runs with lognormals as priors with different relative uncertainty $\rho$ and then see to roughly which value of $\rho$ our prior corresponds by comparing the information (i.e. similar information as our prior for $\rho = \rho^*$ $\rightarrow$ our prior is like $\rho^*\%$ uncertainty)

# + active=""
#                file  new  P Q best joint_prob
# 1: awb/arctic_a0094 TRUE  7 5 TRUE  0.4325568
# 2: bdl/arctic_a0017 TRUE  5 5 TRUE  0.9999998
# 3: jmk/arctic_a0067 TRUE 10 5 TRUE  0.6395072
# 4: rms/arctic_a0382 TRUE 10 5 TRUE  0.9989529
# 5: slt/arctic_b0041 TRUE  7 5 TRUE  0.9999823

# +
# %pylab inline
import analyze
import aux

def do(file, new, order, **kwargs):
    data = aux.get_data(file, 11000)
    hyper = aux.get_hyperparameters()
    return analyze.analyze(new, order, data, hyper, **kwargs)


# -

# ## Analyze "sure-thing" files

# + active=""
#                file  new  P Q best joint_prob
# 2: bdl/arctic_a0017 TRUE  5 5 TRUE  0.9999998
# 4: rms/arctic_a0382 TRUE 10 5 TRUE  0.9989529
# 5: slt/arctic_b0041 TRUE  7 5 TRUE  0.9999823
# -

# ### `bdl/arctic_a0017` (aka. the golden example)
#
# - Well-resolved B1-B4 and F1-F4
# - Splitting of F1 into well-resolved doublet
# - Good glottal flow estimates

full = do('bdl/arctic_a0017', True, (5,5))

# ### `slt/arctic_b0041`
#
# - Well-resolved B1-B3 and F1-F3
# - Splitting of F1 and F2 into well-resolved doublets
# - Well-behaved trend

full = do('slt/arctic_b0041', True, (7,5))

# ### `rms/arctic_a0382`
#
# - Well-resolved B1-B3 and F1-F3
# - Splitting of F1 and F2 into doublets of which the latter is well-resolved
# - The trend $(P=10)$ has a strong low-frequency component of about 300 Hz. This component is also ignored (i.e. not labeled as a formant) in the best `new=False` model. It looks like we need `Q=6` or more for this data to make the best `P` smaller and to pick up this low-frequency component.
# - PDR abnormally high; around zero dB

full = do('rms/arctic_a0382', True, (10,5))

# ## Analyze non "sure-thing" files

# + active=""
#                 file  best  P Q rel_prob
#  1: awb/arctic_a0094  TRUE  7 5       43
#  2: awb/arctic_a0094 FALSE  8 5       37
#  3: awb/arctic_a0094 FALSE  9 5       19
#  7: jmk/arctic_a0067  TRUE 10 5       64
#  8: jmk/arctic_a0067 FALSE  9 5       36
#  9: jmk/arctic_a0067 FALSE  6 5        0
# -

# ### `awb/arctic_a0094`
#
# - Well-resolved B1, B2 and F1, F2
# - Splitting of F1 and F2 into a well-resolved doublet and triplet, resp.
# - Good glottal flow estimate
# - Extremely high SNR
# - Next two most probable model have `best=False`

full = do("awb/arctic_a0094", True, (7,5))

# ### `jmk/arctic_a0067`
#
# - Well-resolved B1-B3 and F1-F3
# - Splitting of F3 into reasonable well-resolved triplet
# - Acceptable trend, even though `P=10`
# - Extremely high SNR, Low PDR
# - Next most probable model has `best=False`

full = do("jmk/arctic_a0067", True, (10,5))


