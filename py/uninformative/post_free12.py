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

# # Post analysis of **free** VTRs nested sampling runs (with 12 poles)
#
# These are "bonus" results, not used in the paper. We use `hyper_free12.py` now.
#
# ## Model order posterior
#
# <img src="post/model_order_posterior_free12.png" alt="2D posterior of the model orders given `new` and file" style="height: 400px;"/>
#
# ## Trend order posterior
#
# <img src="post/trend_order_posterior_free12.png" style="height: 300px;"/>
#
# ## Conclusions
#
# This is a slightly different run from `hyper_free.py`. Allowed Q is in [0, 12]
# and the bounds for the bandwidths and lower bound for R1 are slightly less
# permissive. Comparing with `hyper_free.py`, we may note that:
#
#   - The new prior does its job very well
#   - Generally very good fits; in 3 out of 5 cases the spectrum expansion
#     stops before $Q$ reaches its maximum (i.e. 12)
#   - We obtain consistent results for all VTRs and formant estimates
#   - All MAP model orders have $P \leq 9$ but "little" and "until" have
#     their spectra further resolved until $Q = 12$
#   - F5 resolved in "little"
#
# We decide to go with `hyper_free.py` because it is easier to present. We
# conclude that the results are generally consistent for any $Q$ higher than,
# say, 6, and that one may choose higher value of $Q$ (e.g. $Q = 16$) depending
# on ones purpose. The SNR will not change appreciably after some treshold
# of $Q$ is reached, which is generally smaller than 10 (but not always, as
# in the case of "until", where we reached a +4 dB increase by allowing Q = 12).
# It is probably harmless to just set $Q$ to a high value (e.g. $Q = 16$) and
# infer formants and GF from that model.

# +
# %pylab inline
import analyze
from hyper_free12 import get_data, get_hyperparameters
from plot import show_residuals

def do(file, new, P=None, Q=None, **kwargs):
    data = get_data(file)
    hyper = get_hyperparameters()
    if P is None:
        return analyze.analyze_average(new, Q, data, hyper, **kwargs)
    else:
        order = (P, Q)
        return analyze.analyze(new, order, data, hyper, **kwargs)


# -

# ## `that`

a = do('bdl/arctic_a0017', True, Q=6)

# ## `until`

a = do('slt/arctic_b0041', True, Q=12)

# ## `little`

a = do('rms/arctic_a0382', True, Q=12)

# ## `you`

a = do("jmk/arctic_a0067", True, Q=10)

a = do("jmk/arctic_a0067", True, Q=8)

# ## `shore`

a = do('awb/arctic_a0094', True, Q=8)
