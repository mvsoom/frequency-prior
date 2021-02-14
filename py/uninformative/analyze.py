import numpy as np
import gvar
import copy
import dynesty
from dynesty import utils as dyfunc

import plot
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
    num_resample,
    num_freq
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

def get_analysis(
    new,
    order,
    data,
    hyper,
    num_resample,
    num_freq
):
    P, Q = order
    
    # Get posterior samples of poles
    results = driver_script.run_nested(new, P, Q, data, hyper)
    
    # Calculate pole estimates
    estimates = parameter_estimates(results)
    
    # Get complete posterior samples (i.e. amplitudes, poles and sigma)
    complete_samples, complete_logwts = complete.complete_samples(order, data, results)

    # Calculate complete estimates and SNR estimates
    complete_estimates = parameter_estimates((complete_samples, complete_logwts))
    SNR, bs_SNR_pitch_periods = SNR_estimates(complete_estimates, order, data)
    
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
    
    # Get power spectrum of data and posterior of power spectrum of VT impulse response
    freqs = complete.freqspace(num_freq, data[0])
    spectrum = np.mean(spectra, axis=0) # Average spectra over pitch periods
    
    return {
        'new': new,
        'order': order,
        'data': data,
        'hyper': hyper,
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
        'PDR': PDR,
        'spectra': spectra,
        'freqs': freqs,
        'spectrum': spectrum # Power spectrum in dB averaged over pitch periods
    }

def print_analysis(a):
    results = a['results']
    print("Log Z =", logz(results))
    print("Information (nats) =", results['information'][-1])
    
    estimates = a['estimates']
    P, Q = a['order']; del P
    print("Bandwidths estimates (Hz) =", estimates[:Q])
    print("Frequency estimates (Hz) =", estimates[Q:])
    
    print("Approximate SNR (dB) =", a['SNR'])
    print("Approximate amplitude SNR per pitch period (dB) =")
    for bs in a['bs_SNR_pitch_periods']: print(bs)
    
    print("Periodic to data power ratio PDR (dB) =", a['PDR'])

def analyze(
    new,
    order,
    data,
    hyper,
    num_resample=2000,
    num_freq=250,
    dyplots_kwargs = {},
    modelplots_kwargs = {},
    spectrumplots_kwargs = {}
):
    """Analyze a single (P, Q) model"""
    a = get_analysis(new, order, data, hyper, num_resample, num_freq)
    print_analysis(a)
    
    # Show nested sampling plots
    plot.show_dyplots(a['results'], **dyplots_kwargs)
    
    # Plot components, model function and data
    plot.show_modelplots(data, a['trends'], a['periodics'], a['fs'], **modelplots_kwargs)
    
    # Plot power spectrum of data and inferred impulse response
    plot.show_spectrumplot(data, a['spectrum'], a['freqs'], **spectrumplots_kwargs)
    
    return a

def analyze_average(
    new,
    Q,
    data,
    hyper,
    P_max = 10,
    p_cutoff = .01,
    **analyze_kwargs
):
    """Analyze models Q averaged over trend order P"""
    def get_logz(P):
        results = driver_script.run_nested(new, P, Q, data, hyper)
        return gvar.mean(logz(results))

    lz = [get_logz(P) for P in range(0, P_max+1)]
    return lz