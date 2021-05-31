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

# # Post analysis of **free** VTRs nested sampling runs (with 10 poles)
#
# Cases are ordered in the same way as the original runs (referred to as the OA, "original analysis").
#
# We use `hyper_free.py`, not `hyper_free12.py`.
#
# Note that the variation in the $\log Z$ estimates of the nested sampling runs is large enough such that different runs may give *different* "sure-thing" hypotheses (i.e. model orders with a probability of 1). In other words, the MAP model orders may not be reproducible across different nested sampling runs. Although this phenomenon may seem disturbing,
#
# - It is actually completely consistent with nested sampling theory: one run gives you only one point estimate of $\log Z$ with an estimated error bar -- proper error bars need multiple nested sampling runs
#
# - More importantly: different MAP model orders have actually only small impact on the end results: the inferred VTRs, formants, "glottal flow", and spectral features. The model is robust against misspecification of the model order, an effect which is also noted in Bretthorst (1988).
#
# ## Model order posterior
#
# <img src="post/model_order_posterior_free.png" alt="2D posterior of the model orders given `new` and file" style="height: 400px;"/>
#
# ## Trend order posterior
#
# <img src="post/trend_order_posterior_free.png" style="height: 300px;"/>
#
# ## Conclusions
#
# - Allowing higher $Q$ makes the model expand more and more intricate structure, which may or may not be useful for the application at hand. (Note that our representation of the transfer function by a sum of decaying sinusoids allows for poles *and zeros*.) As a result, PDRs approach 0 dB.
#
# - The trend order $P$ shifts away from its maximum value ($P=10$) and is always well-determined, which shows that the detrending concept is sound when $Q$ is allowed to be relatively large. Allowing higher $Q$ enabled the "absorption" of frequencies by poles instead of trends with high value of $P$. This is in contrast with the original experiments where $Q \leq 5$.
#
# - The overall behavior shows again just how uniformative our prior is: the given expectation values only constrain the prior extremely weakly.
#
# - The higher the lower bound for the bandwidth, the less the model will be inclined to split up peaks -- since higher bandwidth means wider peaks. Next to decreasing the value of $Q$ and adding extra moment information to our prior, this parameter can thus be used to get more "formant-like" measurements, i.e. to get 3 or 4 smoothed peaks (representing F1, F2, etc.) instead of that annoying peak-splitting behaviour.
#
#   But it is important to remember that according to the model and given the assumptions made about the noise, trend, system linearity, etc., these peaks are better modeled as (physical) VTRs rather than smoothed simplifying formants. In addition, all but one models stop at a value of $Q < 10$, suggesting that the peak splitting process is well-defined as it stops until the residuals can be modeled sufficiently well as white noise. Furthermore, we don't get "shaping poles" with large bandwidths -- so we are not really expanding the spectrum (otherwise the model would always prefer $(P=10,Q=10)$); something more sophisticated is going on.
#
# - In 3 of 5 cases (`little`, `you`, `shore`) an extra and well-resolved formant was discovered. These are the cases where this analysis is preferred over the OA.

# +
# %pylab inline
import analyze
from hyper_free import get_data, get_hyperparameters
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
#
# There are two equally plausible hypotheses: the low frequency content in the data is either part of the VT transfer function (`Q=7`) , or part of the trend (`Q=6`). We will only discuss the latter in this list.
#
# - F1 and F2 both split in doublets. The doublets for F1 are well-resolved. F2 is split in two very sharp peaks (bandwidth about 10 Hz, which is the lower bound) which are very close to each other. This seems like an artifact resulting from the very tolerant bandwidth bounds, but runs from `hyper_free12` confirmed the very small bandwidth. It doesn't happen, though, for the original `(new=True, Q=5)` model, which has comparable SNR and a $\log Z$ difference of only 6.
#
# - F3 and F4 are extremely well resolved.
#
# - Other than the splitting of F2, this model is equivalent to the original `(new=True, Q=5)` model with only slightly better SNR and error bars on the parameters and trend.

a = do('bdl/arctic_a0017', True, Q=6) # p(Q=6|data) ~ 39%

a = do('bdl/arctic_a0017', True, Q=7) # p(Q=7|data) ~ 61%

# ## `until`
#
# - The structure in the noise shows that further poles are needed for its expansion -- the SNR is only about 18 dB, which is only +2 dB comparing to OA.
#
# - Presence of very low frequency pole. F1 split in quartet, F2 triplet, F3 doublet. Most frequencies and bandwidths are well-resolved, hence the high information compared to OA.
#
# - The 10 poles have been used to model the very intricate structure of the spectrum, including an (implicit) zero at around 2300 Hz and a very low resonance.
#
# - There is a lot of uncertainty on the trend offset.
#
# This is the case for which the `hyper_free12.py` model achieves improvements:
#
# - SNR can be improved to 22 dB with 12 poles
# - Low-frequency pole disappears
# - "Glottal flow" estimate drastically improves

a = do('slt/arctic_b0041', True, Q=10)

show_residuals(a)

# ## `little`
#
# - Much higher $\log Z$ and information compared to OA, SNR +4 dB
#
# - Better behaved trend and acceptable "glottal flow" estimate. OA's problem of low-frequency component in trend has been solved.
#
# - Discovered F4! OA was limited to F3. Also resolved awkward shoulder in OA's F1 as a tiny extra peak.
#
# - Splitting of F1 into a doublet, F2 into a triplet, both moderately resolved. F3 and F4 well resolved.

a = do('rms/arctic_a0382', True, Q=8)

# ## `you`
#
# - OA's ambiguity in $P=9,10$ resolved to $P=7$. Similar trend and "GF" although more uncertain compared to OA.
#
# - Very high SNR, +4 dB compared to OA. Intricate structure of the transfer function including a zero at about 1500 Hz.
#
# - Discovered F4! OA was limited to F3.
#
# - F1 split in broad triplet. F2 split in well resolved triplet (as in OA). F3 and F4 very well resolved.

a = do("jmk/arctic_a0067", True, Q=8)

show_residuals(a)

# ## `shore`
#
# - OA's ambiguity in $P=7,8,9$ resolved to a mere $P=5$. Low-frequency component of trend in OA has been absorbed, leaving an improved trend and "glottal flow" estimate, though the latter is uncertain because of uncertainty in the amplitude of the trend offset.
#
# - *Extremely* high SNR of 40 dB, which is comparable to the difference in sound level between normal conversation and a chain saw. +7 dB compared to OA.
#
# - Discovery of F3! OA was limited to F2.
#
# - F1 and F2 are very broad peaks. F1 split in quartet, F2 split in triplet/quartet. Both with broad marginal posteriors of the bandwidths and frequencies involved. F3 well resolved.

a = do('awb/arctic_a0094', True, Q=9)

show_residuals(a)
