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

def parameter_estimates(results, return_gvars=True):
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    
    # Compute weighted mean and covariance.
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    estimates = gvar.gvar(mean, cov) if return_gvars else mean
    return estimates

def analyze(results, ylim_quantiles=(0,.99), show_runplot=False):
    estimates = parameter_estimates(results)
    
    print("Log Z =", logz(results))
    print("Information =", results['information'][-1])
    print('Estimates =', estimates)
    #print('Full covariance =')
    #pprint(cov)
    
    if show_runplot: dyplot.runplot(results)
    p, _ = dyplot.traceplot(results, show_titles=True, verbose=True, ylim_quantiles=ylim_quantiles)
    p.tight_layout() # Somehow this still outputs to Jupyter lab
    
    dyplot.cornerplot(results)

def resample_results(results):
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    
    new = copy.deepcopy(results)
    new.samples = dyfunc.resample_equal(samples, weights)
    new.logwt = np.log(np.ones(len(weights))/len(weights))
    return new