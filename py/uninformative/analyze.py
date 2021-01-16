import numpy as np
import gvar
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

def analyze(results, show_runplot=True):
    estimates = parameter_estimates(results)
    
    print("Log Z =", logz(results))
    print("Information =", results['information'][-1])
    print('Estimates =', estimates)
    #print('Full covariance =')
    #pprint(cov)
    
    if show_runplot: dyplot.runplot(results)
    p, _ = dyplot.traceplot(results, show_titles=True, verbose=True)
    p.tight_layout() # Somehow this still outputs to Jupyter lab
    dyplot.cornerplot(results)