import numpy as np
import gvar
import copy
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

def logz(results):
    lz = results['logz'][-1]
    H = results['information'][-1]
    return gvar.gvar(lz, np.sqrt(H/results.nlive))

def parameter_estimates(results, return_gvars=True):
    samples = results.samples  # samples
    weights = np.exp(results.logwt - results.logz[-1])  # normalized weights
    
    # Compute weighted mean and covariance.
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    estimates = gvar.gvar(mean, cov) if return_gvars else mean
    return estimates

def analyze(results, ylim_quantiles=(0,.99)):
    estimates = parameter_estimates(results)
    
    print("Log Z =", logz(results))
    print("Information =", results['information'][-1])
    print('Estimates =', estimates)
    #print('Full covariance =')
    #pprint(cov)
    
    dyplot.runplot(results)
    p, _ = dyplot.traceplot(results, show_titles=True, verbose=True, ylim_quantiles=ylim_quantiles)
    p.tight_layout() # Somehow this still outputs to Jupyter lab
    
    dyplot.cornerplot(results)

def resample_results(results):
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    
    new = copy.deepcopy(results)
    new.samples = dyfunc.resample_equal(samples, weights)
    new.logwt = np.repeat(-np.log(len(weights)), len(weights))
    return new