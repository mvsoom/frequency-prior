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
# ## Model order posterior
#
# <img src="post/model_order_posterior.png" alt="2D posterior of the model orders given `new` and file" style="height: 600px;"/>
#
# ## Trend order posterior
#
# <img src="post/trend_order_posterior.png" style="height: 400px;"/>
#
# ## Conclusions
#
# - `new=False` mildly dominates but $\log Z$ values are equal to `new=True` models within error bounds
#
# - New prior cannot evoke formant behaviour where in original analysis there was VTR behavior: inductive bias of informative prior is too weak
#
# - New prior slightly *less* informative than Jeffreys bounds

# +
# %pylab inline
import analyze
import aux
from plot import show_residuals

def do(file, new, P=None, Q=None, **kwargs):
    data = aux.get_data(file, 11000)
    hyper = aux.get_hyperparameters()
    if P is None:
        return analyze.analyze_average(new, Q, data, hyper, **kwargs)
    else:
        order = (P, Q)
        return analyze.analyze(new, order, data, hyper, **kwargs)


# -

# ## Analyze "sure-thing" files

# ### `that` (aka. the golden example)
#
# - `new=False` preferred 3:1 but slightly more informative (i.e. that model had less gained information going from prior to posterior)
# - Formant behavior
# -  `new=False` multimodal behavior

a = do('bdl/arctic_a0017', True, Q=3)

a = do('bdl/arctic_a0017', False, Q=3)

# ### `until`
#
# - `new=False` preferred 4:1
# - VTR behavior (F1 doublet)

a = do('slt/arctic_b0041', True, Q=3)

a = do('slt/arctic_b0041', False, Q=3)

# ### `little`
#
# - `new=False` preferred 2:1 but slightly more informative
# - VTR behavior (F2 doublet)

a = do('rms/arctic_a0382', True, Q=3)

a = do('rms/arctic_a0382', False, Q=3)

# ### `you`
#
# - `new=False` preferred 3:1 but slightly more informative
# - Formant behavior

a = do("jmk/arctic_a0067", True, Q=3)

a = do("jmk/arctic_a0067", False, Q=3)

# ### `shore`
#
# - `new=True` strongly preferred
# - VTR behavior (F2 doublet)

a = do("awb/arctic_a0094", True, Q=3)

# Try particular order
a = do("awb/arctic_a0094", False, Q=2)
