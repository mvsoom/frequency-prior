"""See complete.md for the logic behind completing the posterior samples"""
import numpy as np
from scipy.stats import invgamma, multivariate_normal
import model

import joblib
memory = joblib.Memory('cache', verbose=0)

def sample_sigma(alpha, beta):
    ig = invgamma(alpha, scale=beta)
    z = ig.rvs()
    lpz = ig.logpdf(z)
    
    sigma = np.sqrt(z)
    lp = lpz + np.log(2*sigma) # Add log Jacobian of transformation
    
    return sigma, lp

def sample_bs(b_hats, gs, sigma):
    bs, lps = [], []
    for b_hat, g in zip(b_hats, gs):
        # TODO: We don't need to calculate the inverse of g -- see complete.md
        cov = sigma**2*np.linalg.inv(g)
        mvn = multivariate_normal(mean=b_hat, cov=cov, allow_singular=True)
        b = mvn.rvs()
        bs += [b]
        lps += [mvn.logpdf(b)]
    
    return bs, lps

@memory.cache
def complete_samples(order, data, results, n_jobs=-1):
    samples, logwt = results.samples, results.logwt
    nu, _ = model.order_factors(order, data, np.nan)
    
    def complete_sample(x, w):
        b_hats, gs, chi2s = [], [], []

        for (t, d) in zip(data[1], data[2]):
            G = model.eval_G(t, x, order)

            b_hat, chi2, rank, s = np.linalg.lstsq(G, d, rcond=None)
            g = G.T @ G
            
            b_hats += [b_hat]
            gs += [g]
            chi2s += [float(chi2)]
        
        sigma, lp = sample_sigma(nu/2, np.sum(chi2s)/2)
        bs, lqs = sample_bs(b_hats, gs, sigma)
        
        complete_sample = np.hstack([*bs, x, [sigma]])
        complete_logwt = w + lp + np.sum(lqs)
        
        return complete_sample, complete_logwt

    with joblib.Parallel(n_jobs=n_jobs) as parallel:
        zipped = parallel(
            joblib.delayed(complete_sample)(x, lw) for x, lw in zip(samples, logwt)
        )

    complete_samples, complete_logwts = zip(*zipped)
    return np.array(complete_samples), np.array(complete_logwts)