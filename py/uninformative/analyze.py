import numpy as np
import gvar
import copy
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
import matplotlib.pyplot as plt

import driver_script
import complete

def logz(results):
    lz = results['logz'][-1]
    # Use crude estimate of standard deviation of logz from @Skilling2006
    # instead of the one supplied by dynesty -- the latter can get awkwardly
    # large.
    #sd = results['logzerr'][-1]
    H = results['information'][-1]
    N = results['nlive']
    sd = np.sqrt(H/N)
    return gvar.gvar(lz, sd)

def exp_and_normalize(logw):
    # We don't normalize with log Z because complete posterior samples can
    # have log densities much larger than log Z, leading to large weights.
    nw = np.exp(logw - np.max(logw))
    nw /= sum(nw)
    return nw

def parameter_estimates(results_or_samples, return_gvars=True):
    try:
        samples = results_or_samples.samples
        logwt = results_or_samples.logwt
    except AttributeError:
        samples, logwt = results_or_samples
    
    weights = exp_and_normalize(logwt)
    
    # Compute weighted mean and covariance.
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    estimates = gvar.gvar(mean, cov) if return_gvars else mean
    return estimates

def _log10(gvar):
    return np.log(gvar)/np.log(10)

def SNR_estimates(complete_estimates, order, data):
    """Calculate SNR for the model function f and individual amplitudes
    
    Note that we use approximations. In the case of global SNR, the
    correct approach would be to calculate the SNR
    
        SNR = 10 log10[f.T @ f/N sigma²]
    
    for each complete sample (bs, x, sigma) -- where we calculate f from bs
    and x as is done in complete.sample_trends_and_periodics() -- and
    average over the thusly acquired SNR samples to get the final result.
    
    In practice, however, if the fit is reasonable, a very good approximation
    is simply taking the empirical SNR with the noise power set to the
    estimate of sigma². This is what we do here.
    
    In the case of the amplitude SNRs, we should follow the same approach.
    Instead we just use the gvar approximation of dividing the Gaussian
    representations of the amplitudes and sigma, which holds if the stdev
    is sufficiently small. This is not the case for some amplitudes, and
    very wide stdevs are returned sometimes. Therefore the resulting
    SNRs are only accurate if their stdevs are small. If this is not the
    case, this probably means the actual posterior distrubution of the
    SNR is badly approximated by a Gaussian (too wide/multimodality/etc.).
    """
    P, Q = order
    n = len(data[1])
    m = P + 2*Q
    bs, x, sigma = np.split(complete_estimates, [n*m, -1])
    del x
    
    d = np.concatenate(data[2])
    SNR = 10*_log10(d.T @ d/(len(d)*sigma**2))
    
    bs_SNR = 10*_log10(bs**2/sigma**2)
    bs_SNR_pitch_periods = np.split(bs_SNR, n)

    return SNR, bs_SNR_pitch_periods

def PDR_estimate(ds, periodics):
    d = np.concatenate(ds)
    p = np.concatenate(periodics)

    return 10.*_log10(np.sum(p**2)/np.sum(d**2))

def model_function_estimates(
    complete_samples,
    complete_logwts,
    order,
    data,
    num_resample=2000,
    num_freq=1000
):
    samples = complete.sample_components_and_spectrum(
        complete_samples,
        complete_logwts,
        num_resample,
        order,
        data,
        num_freq
    )
    
    trends_samples, periodics_samples, spectra_samples = samples

    n = len(data)
    fs_samples = [trends_samples[j] + periodics_samples[j] for j in range(n)]

    def to_gvar(a):
        mean = np.mean(a, axis=0)
        cov = np.cov(a, rowvar=False)
        return gvar.gvar(mean, cov)

    trends = [to_gvar(a) for a in trends_samples]
    periodics = [to_gvar(a) for a in periodics_samples]
    fs = [to_gvar(a) for a in fs_samples]
    spectra = [to_gvar(a) for a in spectra_samples]
    
    return trends, periodics, fs, spectra

def resample_equal(samples, logwt, n):
    weights = exp_and_normalize(logwt)
    samples = dyfunc.resample_equal(samples, weights)
    i = np.random.choice(len(weights), size=n, replace=False)
    return samples[i,:]

def analyze(
    new,
    order,
    data,
    hyper,
    complete_analysis=True,
    plot=True,
    num_resample=2000,
    num_freq=250,
    dyplots_kwargs = {},
    modelplots_kwargs = {},
    spectrumplots_kwargs = {}
):
    P, Q = order
    
    # Get samples
    results = driver_script.run_nested(new, P, Q, data, hyper)
    
    # Calculate estimates
    estimates = parameter_estimates(results)
    
    print("Log Z =", logz(results))
    print("Information (nats) =", results['information'][-1])
    
    print("Bandwidths estimates (Hz) =", estimates[:Q])
    print("Frequency estimates (Hz) =", estimates[Q:])
    
    # Show nested sampling plots
    if plot: show_dyplots(results, **dyplots_kwargs)
    
    # Bail out early if possible
    if not complete_analysis:
        return {'results': results, 'estimates': estimates}
    
    # Get complete samples
    complete_samples, complete_logwts = complete.complete_samples(order, data, results)

    # Calculate complete estimates and SNR estimates
    complete_estimates = parameter_estimates((complete_samples, complete_logwts))
    SNR, bs_SNR_pitch_periods = SNR_estimates(complete_estimates, order, data)
    
    print("Approximate SNR (dB) =", SNR)
    print("Approximate amplitude SNR per pitch period (dB) =")
    for bs in bs_SNR_pitch_periods: print(bs)
    
    # Calculate trend and periodic components using resampling
    trends, periodics, fs, spectra = model_function_estimates(
        complete_samples,
        complete_logwts,
        order,
        data,
        num_resample=num_resample,
        num_freq=num_freq
    )
    
    PDR = PDR_estimate(data[2], periodics)
    print("Periodic to data power ratio PDR (dB) =", PDR)
    
    # Plot components, model function and data
    if plot: show_modelplots(data, trends, periodics, fs, **modelplots_kwargs)
    
    # Plot power spectrum of data and inferred impulse response
    freqs = complete.freqspace(num_freq, data[0])
    spectrum = np.mean(spectra,axis=0) # Average spectra over pitch periods
    
    if plot: show_spectrumplot(data, spectrum, freqs, **spectrumplots_kwargs)
    
    return {
        'results': results,
        'estimates': estimates,
        'complete_samples': complete_samples,
        'complete_logwts': complete_logwts,
        'complete_estimates': complete_estimates,
        'SNR': SNR,
        'bs_SNR_pitch_periods': bs_SNR_pitch_periods,
        'trends': trends,
        'periodics': periodics,
        'fs': fs,
        'PDR'
        'spectra': spectra,
        'freqs': freqs,
        'spectrum': spectrum # Power spectrum in dB averaged over pitch periods
    }

def get_labels(results):
    Q = int(results.samples.shape[1] / 2)
    return [f"$B_{i+1}$" for i in range(Q)] + [f"$F_{i+1}$" for i in range(Q)]

def show_dyplots(results, ylim_quantiles=(0,.99), trace_only=True):
    if trace_only is None: return

    if not trace_only: dyplot.runplot(results)

    # This uses a locally modified version of dyplot.traceplot(). The ylim_quantiles
    # are used to reject samples with a likelihood of zero, i.e. samples with
    # formant frequencies larger than fs/2. There are other ways of dealing with this,
    # such as resampling or sifting out samples with logl = -1e300.
    p, _ = dyplot.traceplot(
        results, show_titles=True, ylim_quantiles=ylim_quantiles, labels = get_labels(results)
    )
    p.tight_layout() # Somehow this still outputs to Jupyter lab

    if not trace_only: dyplot.cornerplot(results)

def show_modelplots(
    data,
    trends,
    periodics,
    fs,
    num_posterior_samples=25,
    offset=2,
    figsize=(12,2)
):
    def ugly_hack(data, d):
        t = data[1][0]
        dt = t[1] - t[0]
        return np.arange(len(d))*dt*1000 # (msec)
    
    d = np.concatenate(data[2])
    t = ugly_hack(data, d)
    trend = np.concatenate(trends)
    periodic = np.concatenate(periodics)
    f = np.concatenate(fs)
    
    def samples(g):
        s = [gvar.sample(g) for i in range(num_posterior_samples)]
        return np.array(s).T

    def plot_data(i):
        plt.plot(t, d - i*offset, '--', color='black')

    def plot_samples(g, i, color='black', alpha=1/num_posterior_samples):
        plt.plot(t, samples(g) - i*offset, color=color, alpha=alpha)

    width, height = figsize
    plt.figure(figsize=(width, height*3))
    plt.title('Data vs. posterior samples of the model function and its components')
    plt.xlabel('time [msec]')
    plt.ylabel('amplitude [a.u.]')

    # Plot data vs. full model function
    plot_data(0)
    plot_samples(f, 0)
    
    # Plot components
    plot_data(1)
    plot_samples(trend, 1)
    
    plot_data(2)
    plot_samples(periodic, 2)
    
    plt.show()
    
    # Plot normalized glottal flow
    plt.figure(figsize=(width, height))
    plt.title('Data vs. posterior samples of the "glottal flow"')
    plt.xlabel('time [msec]')
    plt.ylabel('amplitude [a.u.]')
    
    gf = np.cumsum(trend)
    gf_mean = gvar.mean(gf)
    scale = gf_mean.max() - gf_mean.min()
    
    plot_data(0)
    plot_samples(gf/scale, 0)

    plt.show()

def show_spectrumplot(
    data,
    spectrum,
    freqs,
    n_pad=None,
    num_posterior_samples=25,
    figsize=(12,4)
):
    plt.figure(figsize=figsize)
    plt.title('Spectrum of data vs. posterior samples of impulse response spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Spectral power (dB)')

    # Calculate power spectrum of data with correct (dt scaling).
    # See ./FFT_scaling.ipynb for details.
    d = np.concatenate(data[2])
    dt = 1/data[0]
    D = np.fft.rfft(d, n_pad)*dt
    D_freq = np.fft.rfftfreq(n_pad if n_pad else len(d), dt)

    plt.plot(D_freq, 20*np.log10(np.abs(D)), '--', color='black')
    
    # Plot posterior samples of pitch-period-averaged power spectrum (already in dB)
    def samples(g):
        s = [gvar.sample(g) for i in range(num_posterior_samples)]
        return np.array(s).T

    plt.plot(freqs, samples(spectrum), color='black', alpha=1/num_posterior_samples)
    
    plt.show()