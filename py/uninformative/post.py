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
# - The new prior and highest $Q$ `(new=True,Q=5)` dominates (100%) for every file, which enables easy model averaging. (Averaging models over $Q$ gives the problem of combining the right frequencies with each other, since the frequencies can shift anywhere; this is not a problem for the `new=False` models where the frequencies live in the intervals specified by the Jeffreys priors.)
#
# - In contrast, the MAP $Q$ values for the `new=False` models are between two (`awb`) to four (`bdl`) and three for the others.
#
# - As before, we always find evidence of a trend (`P > 0`). Such low-frequency components imply that the unaltercated Fourier transform magnitude spectrum is a suboptimal estimator (Van Soom 2019a). "Unaltercated" here means without windowing, detrending, and other "ad-hoc" measures.
#
# - The splitting of peaks into doublets and triplets is also seen in Bretthorst (1988) and may indicate physical effects; or it could be artifacts stemming from inappropriate assumptions such as Lorentzian decay (as implied by the assumed LTI system theory). It is a general feature of our model (the likelihood function) and also happens when allowed by the old prior -- but in an inconvenient way: see "All old priors have inconveniences" below. Our model is thus a VTR model by nature, and using it to (frustrating it into) estimate formants is not trivial. The fact that specifiying (Jeffreys) bounds for VTRs without mode hopping or other nuisances is hard is a problem that our new prior solves.
#
# - Ways to properly turn the model into a formant rather than a VTR estimator could be achieved by:
#
#   * Changing the likelihood: expand the decay modes into polynomials or another parametrization as in (Bretthorst 1988) to model the broad peaks -- this effectively gets us beyond LTI poles and zeros (but still LTI; just not expressible with a rational transfer function).
#   * And simultaneously using a prior with *correlations* between the $u_j$ to make sure that formant frequencies don't get too close to each other. Our prior cannot do this; it is too uninformative.
#
# - The doublets and triplets also happen in LPC and could be interpreted as "only" spectrum shaping factors. However, the following facts seem to point toward physical existence of these multiplets:
#   * Their bandwidths are well-behaved (e.g. around 50 or 100 Hz); shaping formants usually have broad bandwidths (i.e. very broad peaks)
#   * The multiplet frequencies cluster around peaks; shaping formants are typically more between peaks
#   * The multiplet resolving behavior is quite particular to the data, both in number of split components and the formant peak which is split. For example, `jmk/arctic_a0067` has F2 split in a triplet.
#
# -----
#
# - All old priors have inconveniences in one way or another: **mode hopping** (upsetting mean ± std estimates); **bound frustration** because the model wants to resolve a VTR leaning against a bound derived from formant measurements;  **missing higher formants** because the model turned the formant into a VTR; **underestimating formant bandwidths** because extra VTRs are needed to resolve a broad formant.
#
# - All formant estimates agree between old and new prior, except the old prior misses a formant or underestimates the width of a formant. The formant estimator is thus very robust.
#
# - All formant estimates agree roughly with [known F1-F2 values](https://en.wikipedia.org/wiki/Formant#/media/File:Average_vowel_formants_F1_F2.png)

# +
# %pylab inline
import analyze
from hyper import get_data, get_hyperparameters
from plot import show_residuals

def do(file, new, P=None, Q=None, **kwargs):
    data = get_data(file, 11000)
    hyper = get_hyperparameters()
    if P is None:
        return analyze.analyze_average(new, Q, data, hyper, **kwargs)
    else:
        order = (P, Q)
        return analyze.analyze(new, order, data, hyper, **kwargs)


# -

# ## Analyze "sure-thing" files

# ### `that` (aka. the golden example)
#
# - Well-resolved B1-B4 and R1-R4
# - Splitting of F1 into well-resolved doublet
# - Good glottal flow estimates
# - Good formant estimates
#
# Old prior:
#
# - No F1 doublet: just 4 formants
# - Mode hopping between F2 and F3, which upsets mean ± std estimates.
# - Formant estimates agree with new prior

a = do('bdl/arctic_a0017', True, Q=5)

a = do('bdl/arctic_a0017', False, Q=4)

# ### `until`
#
# - Well-resolved B1-B3 and R1-R3
# - Splitting of F1 and F2 into well-resolved doublets
# - Well-behaved trend
#
# Old prior:
#
# - Only found 2 formants
# - F1 split in doublet but second frequency leans against its lower bound
# - Formant estimates agree with new prior

a = do('slt/arctic_b0041', True, Q=5)

a = do('slt/arctic_b0041', False, Q=3)

# ### `little`
#
# - Well-resolved B1-B3 and R1-R3
# - Splitting of F1 and F2 into doublets of which the latter is well-resolved
# - The trend $(P=10)$ has a strong low-frequency component of about 300 Hz. This component is also ignored (i.e. not labeled as a formant) in the best `new=False` model. It looks like we need `Q=6` or more for this data to make the best `P` smaller and to pick up this low-frequency component.
# - PDR abnormally high; around zero dB (which is similar to what VTR models can achieve)
#
# Old prior:
#
# - Difficulty discovering F3: odds against F3 are 2:1. Estimate of F3 given $Q=4$ is off by about 5 sigmas
# - F2 doublet well resolved
# - Formant estimates agree with new prior, but F3 is not picked up because it the bump doesn't have enough prominence

a = do('rms/arctic_a0382', True, Q=5)

a = do('rms/arctic_a0382', False, Q=3)

a = do('rms/arctic_a0382', False, Q=4)

# ## Analyze non "sure-thing" files

# ### `you`
#
# - Well-resolved B1-B3 and R1-R3
# - Splitting of F2 into reasonable well-resolved *triplet*; formant estimates continue to work well
# - Acceptable trend
# - Extremely high SNR, Low PDR
# - `P=10` and `P=9` very similar
#
# Old prior
#
# - Peak hopping between R2 and R3, which upsets their mean ± sd estimates
# - F3 *is* in fact resolved, as can be seen in the spectrum. Formant estimates continue to work, but bandwidth of F2 is underestimated due to missing pole in that vicinity
# - Formant estimates agree with new prior

a = do("jmk/arctic_a0067", True, Q=5)

a = do("jmk/arctic_a0067", False, Q=3)

# ### `shore`
#
# - High uncertainty about `P`
# - Well-resolved B2-B5 and R1-R5; B1 is broad
# - Splitting of F1 and F2 into a well-resolved doublet and *triplet*, resp.
# - Good glottal flow estimate
# - Extremely high SNR
# - Next most probable models have higher `P`:
#   * `P=8`: Very similar to `P=7`
#   * `P=9`: Trend has a low-frequency component of about 250 Hz. Model much higher uncertainty and corresponding lower information than best model. **Thus with this example we see that oscillatory behaviour of trend is penalized but probably not enough.**
#   
#   Still the `P`-averaged glottal flow posterior estimate is sharp; the uncertainties do not interfere negatively
#   
# Old model:
#
# - Just expands F1 and F2 in two very clean VTRs (R1 and R2)
# - Formant estimates agree roughly with new prior; F1 is off because width of peak is underestimated

a = do("awb/arctic_a0094", True, Q=5)

a = do("awb/arctic_a0094", False, Q=2)
