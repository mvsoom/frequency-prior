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
# Cases are ordered in terms of niceness; the nicest examples are first.
#
# ## Conclusions
#
# - As before, we always find evidence of a trend (`P > 0`). Such low-frequency components imply that the unaltercated Fourier transform magnitude spectrum is a suboptimal estimator (Van Soom 2019a). "Unaltercated" here means without windowing, detrending, and other "ad-hoc" measures.
#
# - The new prior enables resolving frequency peaks into doublets and triplets (more general: multiplets). The reason we did not see this before (`new=False`) is because of non-overlapping intervals with the old prior, which precluded F2 getting near F1. We could say that the new prior allows to pick up all the *vocal tract resonancies* which are more fine grained than the canonical formants we are used to; in this example we were able to resolve F1 "doublets".
#
# - These doublets and triplets also happen in LPC and could be interpreted as "only" spectrum shaping factors. However, the following facts seem to point toward physical existence of these multiplets:
#   * Their bandwidths are well-behaved (e.g. around 50 or 100 Hz); shaping formants usually have broad bandwidths (i.e. very broad peaks)
#   * The multiplet frequencies cluster around peaks; shaping formants are typically more between peaks
#   * The multiplet resolving behavior is quite particular to the data, both in number of split components and the formant peak which is split. For example, `jmk/arctic_a0067` has F3 split in a triplet.

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
#                 file  best  new  P Q rel_prob
#  1: awb/arctic_a0094  TRUE TRUE  7 5       43
#  2: awb/arctic_a0094 FALSE TRUE  8 5       37
#  3: awb/arctic_a0094 FALSE TRUE  9 5       19
#  7: jmk/arctic_a0067  TRUE TRUE 10 5       64
#  8: jmk/arctic_a0067 FALSE TRUE  9 5       36
#  9: jmk/arctic_a0067 FALSE TRUE  6 5        0
# -

# ### `jmk/arctic_a0067`
#
# - Well-resolved B1-B3 and F1-F3
# - Splitting of F3 into reasonable well-resolved triplet
# - Acceptable trend, even though `P=10`
# - Extremely high SNR, Low PDR
# - Next most probable model is identical but for `P=9`

full = do("jmk/arctic_a0067", True, (10,5))

# ### `awb/arctic_a0094`
#
# - Well-resolved B1, B2 and F1, F2
# - Splitting of F1 and F2 into a well-resolved doublet and triplet, resp.
# - Good glottal flow estimate
# - Extremely high SNR
# - Next most probable models have higher `P`:
#   * `P=8`: Very similar to `P=7`
#   * `P=9`: Trend has a low-frequency component of about 250 Hz. Model much higher uncertainty and corresponding lower information than best model. **Thus with this example we see that oscillatory behaviour of trend is penalized but probably not enough.**

full = do("awb/arctic_a0094", True, (7,5))

# Next most probable model
full = do("awb/arctic_a0094", True, (8,5))

# Next most probable model
full = do("awb/arctic_a0094", True, (9,5))
