import numpy as np
import gvar
import copy
import dynesty
from dynesty import utils as dyfunc
import tabulate

import joblib
memory = joblib.Memory('cache', verbose=0)

import plot
import driver_script
import complete
import formant

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
    nw /= np.sum(nw)
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

# This function can't be cached because the cached gvars "go stale" when
# adding them, as is done in get_analysis_average()
def get_analysis(
    new,
    order,
    data,
    hyper,
    num_resample,
    num_freq
):
    """Analyze a single (P, Q) model"""
    P, Q = order
    
    # Get posterior samples of poles
    results = driver_script.run_nested(new, P, Q, data, hyper)
    
    # Calculate VTR estimates
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
    
    a = {
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
    return a

def print_analysis(a):
    results = a['results']

    print("Log Z =", logz(results))
    print("Information (nats) =", results['information'][-1])

    print("Approximate SNR (dB) =", a['SNR'])
    print("Periodic to data power ratio PDR (dB) =", a['PDR'])
    
    print_pole_table(a['estimates'], 'R')
    
    print("Approximate amplitude SNR per pitch period (dB) =")
    for bs in a['bs_SNR_pitch_periods']: print(bs)
    
    print_formants(a)

def print_formants(a):
    estimates = formant.estimate_formants(a['freqs'], a['spectrum'])
    print_pole_table(estimates, 'F')

def print_pole_table(estimates, symbol=''):
    bandwidths, frequencies = np.split(estimates, 2)
    headers = [f'{symbol}{i}' for i in (1 + np.arange(len(bandwidths)))]
    
    t = tabulate.tabulate([bandwidths, frequencies], headers=headers, tablefmt='fancy_grid')
    print("Bandwidths and frequency estimates (Hz):")
    print(t)

def print_analysis_average(a):
    print("Approximate SNR (dB) =", a['SNR'])
    print("Periodic to data power ratio PDR (dB) =", a['PDR'])
    
    print_pole_table(a['estimates'], 'R')
    
    print_formants(a)

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
    a = get_analysis(new, order, data, hyper, num_resample, num_freq)
    print_analysis(a)
    
    # Show nested sampling plots
    plot.show_dyplots(a['results'], **dyplots_kwargs)
    
    # Plot components, model function and data
    plot.show_modelplots(data, a['trends'], a['periodics'], a['fs'], **modelplots_kwargs)

    # Plot power spectrum of data and inferred impulse response
    plot.show_spectrumplot(data, a['spectrum'], a['freqs'], a['estimates'], **spectrumplots_kwargs)
    
    return a

def get_significant_Ps(new, Q, data, hyper, P_max, p_cutoff):
    def get_logz(P):
        results = driver_script.run_nested(new, P, Q, data, hyper)
        return logz(results)

    Ps = np.arange(P_max+1)
    lz = np.array([get_logz(P) for P in Ps])
    
    p = np.exp(lz - np.max(lz))
    p /= np.sum(p)
    keep = p > p_cutoff
    
    Ps = Ps[keep]
    weights = p[keep]
    weights /= np.sum(weights)

    return list(Ps), list(weights) # Weights contain mean +/- sd

def get_analysis_average(
    new,
    Q,
    data,
    hyper,
    P_max = 10,
    p_cutoff = .05,
    num_resample=2000,
    num_freq=250
):
    """Analyze models Q averaged over trend order P"""
    Ps, weights = get_significant_Ps(new, Q, data, hyper, P_max, p_cutoff)
    
    # Build a list of significant models according to p_cutoff
    models = {
        P: {
            'weight': gvar.mean(weight),
            'a': get_analysis(new, (P, Q), data, hyper, num_resample, num_freq)
        } for P, weight in zip(Ps, weights)
    }
    
    def model_average(prop_key, i=None):
        avg = np.array([0.]) # To be recasted
        for model in models.values():
            w = model['weight']
            x = model['a'][prop_key]
            if i is not None:
                x = x[i]
            
            avg = avg + w*x

        return avg
    
    def model_list_average(prop_key, n):
        return [model_average(prop_key, i) for i in range(n)]
    
    freqs = models[Ps[0]]['a']['freqs']
    
    # Get number of pitch periods
    n = len(data[1])
    
    a = {
        'Ps':        Ps,
        'Q':         Q,
        'weights':   weights,
        'models':    models,
        'estimates': model_average('estimates'),
        'SNR':       model_average('SNR'),
        'trends':    model_list_average('trends', n),
        'periodics': model_list_average('periodics', n),
        'fs':        model_list_average('fs', n),
        'PDR':       model_average('PDR'),
        'spectra':   model_list_average('spectra', n),
        'freqs':     freqs,
        'spectrum':  model_average('spectrum') # Power spectrum in dB averaged over pitch periods
    }
    return a

def average_samples(models):
    samples, logwt = [], []
    for model in models.values():
        w = model['weight']
        a = model['a']
        samples += [a['results']['samples']]
        logwt += [a['results']['logwt'] + np.log(w)]
    
    samples = np.vstack(samples)
    logwt = np.concatenate(logwt)
    weights = exp_and_normalize(logwt)
    return samples, weights

def print_P_table(a):
    headers = [f'P={P}' for P in a['Ps']]
    percents = [f'{100.*w}%' for w in a['weights']]
    
    t = tabulate.tabulate([percents], headers=headers, tablefmt='fancy_grid')
    print(f"Posterior probability prob(P|Q={a['Q']},data):")
    print(t)

def analyze_average(
    new,
    Q,
    data,
    hyper,
    P_max = 10,
    p_cutoff = .05,
    num_resample=2000,
    num_freq=250,
    marginalplots_kwargs = {},
    modelplots_kwargs = {},
    spectrumplots_kwargs = {}
):
    a = get_analysis_average(new, Q, data, hyper, P_max, p_cutoff, num_resample, num_freq)
    
    print_P_table(a)
    
    Ps = a['Ps']
    if len(Ps) == 1:
        # One value of P dominates: run the analysis for the MAP approximation
        order_MAP = (Ps[0], Q)
        a = analyze(
            new,
            order_MAP,
            data,
            hyper,
            num_resample,
            num_freq,
            modelplots_kwargs=modelplots_kwargs,
            spectrumplots_kwargs=spectrumplots_kwargs
        )
        return a
    else:
        print_analysis_average(a)

        # Plot marginalized posteriors for model-averaged samples 
        samples, weights = average_samples(a['models'])
        plot.show_marginalplots(samples, weights, show_titles=True, **marginalplots_kwargs);

        # Plot components, model function and data
        plot.show_modelplots(data, a['trends'], a['periodics'], a['fs'], **modelplots_kwargs)

        # Plot power spectrum of data and inferred impulse response
        plot.show_spectrumplot(data, a['spectrum'], a['freqs'], a['estimates'], **spectrumplots_kwargs)

        return a