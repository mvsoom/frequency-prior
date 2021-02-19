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
# - Our prior for formants is so uninformative that it uncovers more general vocal tract resonances, which depending on the application may be a blessing or a nuisance. The splitting of peaks into doublets and triplets is also seen in Bretthorst (1988) and may indicate physical effects; or it could be artifacts stemming from inappropriate assumptions such as Lorentzian decay (as implied by the assumed LTI system theory).
#
# - `(new=True,Q=5)` dominates (100%) for every file, which enables easy model averaging. (Averaging models over $Q$ gives the problem of combining the right frequencies with each other, since the frequencies can shift anywhere; this is not a problem for the `new=False` models where the frequencies live in the intervals specified by the Jeffreys priors.)
#
# - In contrast, the MAP $Q$ values for the `new=False` models are between two (`awb`) to four (`bdl`) and three for the others.
#
# - As before, we always find evidence of a trend (`P > 0`). Such low-frequency components imply that the unaltercated Fourier transform magnitude spectrum is a suboptimal estimator (Van Soom 2019a). "Unaltercated" here means without windowing, detrending, and other "ad-hoc" measures.
#
# - The new prior enables resolving frequency peaks into doublets and triplets (more general: multiplets). The reason we did not see this before (`new=False`) is because of non-overlapping intervals with the old prior, which precluded F2 getting near F1. We could say that the new prior allows to pick up all the *vocal tract resonancies* which are more fine grained than the canonical formants we are used to; in this example we were able to resolve F1 "doublets".
#
# - These doublets and triplets also happen in LPC and could be interpreted as "only" spectrum shaping factors. However, the following facts seem to point toward physical existence of these multiplets:
#   * Their bandwidths are well-behaved (e.g. around 50 or 100 Hz); shaping formants usually have broad bandwidths (i.e. very broad peaks)
#   * The multiplet frequencies cluster around peaks; shaping formants are typically more between peaks
#   * The multiplet resolving behavior is quite particular to the data, both in number of split components and the formant peak which is split. For example, `jmk/arctic_a0067` has F2 split in a triplet.

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
# - Well-resolved B1-B4 and F1-F4
# - Splitting of F1 into well-resolved doublet
# - Good glottal flow estimates

a = do('bdl/arctic_a0017', True, Q=5)

# ### `until`
#
# - Well-resolved B1-B3 and F1-F3
# - Splitting of F1 and F2 into well-resolved doublets
# - Well-behaved trend

a = do('slt/arctic_b0041', True, Q=5)

# ### `little`
#
# - Well-resolved B1-B3 and F1-F3
# - Splitting of F1 and F2 into doublets of which the latter is well-resolved
# - The trend $(P=10)$ has a strong low-frequency component of about 300 Hz. This component is also ignored (i.e. not labeled as a formant) in the best `new=False` model. It looks like we need `Q=6` or more for this data to make the best `P` smaller and to pick up this low-frequency component.
# - PDR abnormally high; around zero dB (which is similar to what VTR models can achieve)

a = do('rms/arctic_a0382', True, Q=5)

# ## Analyze non "sure-thing" files

# ### `you`
#
# - Well-resolved B1-B3 and F1-F3
# - Splitting of F2 into reasonable well-resolved triplet
# - Acceptable trend, even though `P=10`
# - Extremely high SNR, Low PDR
# - Next most probable model is identical but for `P=9`

a = do("jmk/arctic_a0067", True, Q=5)

# ### `shore`
#
# - Well-resolved B1, B2 and F1, F2
# - Splitting of F1 and F2 into a well-resolved doublet and triplet, resp.
# - Good glottal flow estimate
# - Extremely high SNR
# - Next most probable models have higher `P`:
#   * `P=8`: Very similar to `P=7`
#   * `P=9`: Trend has a low-frequency component of about 250 Hz. Model much higher uncertainty and corresponding lower information than best model. **Thus with this example we see that oscillatory behaviour of trend is penalized but probably not enough.**

a = do("awb/arctic_a0094", True, Q=5)

# Try particular order
a = do("awb/arctic_a0094", True, P=8, Q=5)

# Try particular order
a = do("awb/arctic_a0094", True, P=9, Q=5)
