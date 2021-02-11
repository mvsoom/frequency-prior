import numpy as np
import gvar
import copy
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

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
    
    def log10(gvar): return np.log(gvar)/np.log(10)
    
    d = np.concatenate(data[2])
    SNR = 10*log10(d.T @ d/(len(d)*sigma**2))
    
    bs_SNR = 10*log10(bs**2/sigma**2)
    bs_SNR_pitch_periods = np.split(bs_SNR, n)

    return SNR, bs_SNR_pitch_periods

def model_function_estimates(
    complete_samples,
    complete_logwts,
    order,
    data,
    num_resample=2000
):
    trends_samples, periodics_samples = complete.sample_trends_and_periodics(
        complete_samples,
        complete_logwts,
        num_resample,
        order,
        data
    )

    fs_samples = [trends_samples[j] + periodics_samples[j] for j in range(n)]

    def to_gvar(a):
        mean = np.mean(a, axis=0)
        cov = np.cov(a, rowvar=False)
        return gvar.gvar(mean, cov)

    trends = [to_gvar(a) for a in trends_samples]
    periodics = [to_gvar(a) for a in periodics_samples]
    fs = [to_gvar(a) for a in fs_samples]
    
    return trends, periodics, fs

def analyze(results, ylim_quantiles=(0,.99), trace_only=True):
    estimates = parameter_estimates(results)
    
    print("Log Z =", logz(results))
    print("Information =", results['information'][-1])
    print('Estimates =', estimates)
    #print('Full covariance =')
    #pprint(cov)
    
    if not trace_only: dyplot.runplot(results)
    p, _ = dyplot.traceplot(results, show_titles=True, verbose=True, ylim_quantiles=ylim_quantiles)
    p.tight_layout() # Somehow this still outputs to Jupyter lab
    
    if not trace_only: dyplot.cornerplot(results)

def resample_equal(samples, logwt, n):
    weights = exp_and_normalize(logwt)
    samples = dyfunc.resample_equal(samples, weights)
    i = np.random.choice(len(weights), size=n, replace=False)
    return samples[i,:]