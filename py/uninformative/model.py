import numpy as np
import scipy.special
import dynesty
import time
from sample_ppf import sample_jeffreys_ppf, sample_x_ppf

pi = np.pi
log_pi = np.log(pi)
_loggamma = scipy.special.loggamma
_legendre_basis = np.polynomial.legendre.Legendre.basis

import joblib
memory = joblib.Memory('cache', verbose=1)

@memory.cache
def run_nested(new, order, data, hyper, delta=1., runid=0):
    P, Q = order
    ndim = 2*Q
    
    sampler = dynesty.NestedSampler(
        loglike,
        ptform_new if new else ptform_old,
        ndim=ndim,
        logl_args=(order, data, hyper, delta),
        ptform_args=(order, hyper)
    )
    
    start = time.time()
    sampler.run_nested()
    end = time.time()
    
    res = sampler.results
    res['walltime'] = end - start
    res['runid'] = runid
    return res

def ptform_old(q, order, hyper):
    P, Q = order
    bounds, F = hyper
    
    qb = q[:Q]
    b = sample_jeffreys_ppf(qb, bounds['b'])
    
    qf = q[Q:]
    f = sample_jeffreys_ppf(qf, bounds['f'])
    
    x = np.zeros(2*Q)
    x[:Q] = b
    x[Q:] = f
    return x

def ptform_new(q, order, hyper):
    P, Q = order
    bounds, F = hyper
    
    qb = q[:Q]
    b = sample_jeffreys_ppf(qb, bounds['b'])
    
    qf = q[Q:]
    f = sample_x_ppf(qf, Q, F)
    
    x = np.zeros(2*Q)
    x[:Q] = b
    x[Q:] = f
    return x

def eval_G(t, x, order):
    # Note: unlike original implementation, we don't use scaled variables
    # explicitly. The Legendre polynomials implementation scales time
    # using its domain argument, and the scale factors cancel in the
    # cos() and exp() calls. We just had to take care to include the right
    # trigonometric factors (2*pi) and (pi) for the frequency and bandwidth.
    P, Q = order
    m = P + 2*Q
    G = np.empty((len(t), m))

    # Legendre polynomials of order (0, P-1)
    domain = (t[0], t[-1])
    for j in range(P):
        G[:, j] = _legendre_basis(j, domain)(t)

    # Damped sinusoids
    b = x[:Q]
    f = x[Q:]
    B, F, T = b[None,:], f[None,:], t[:,None]
    
    G[:, P:P+Q] = np.cos(2.*pi*F*T)*np.exp(-pi*B*T)
    G[:, P+Q:] = np.sin(2.*pi*F*T)*np.exp(-pi*B*T)

    return G # (N, m)

def order_factors(order, data, delta):
    n = len(data[0])
    N = sum([len(d) for d in data[0]])
    P, Q = order
    m = P + 2*Q
    
    nu = N - n*m
    logc = -nu*log_pi/2 + _loggamma(nu/2) - n*m*np.log(2*pi*delta**2)/2
    return nu, logc

def loglike(x, order, data, hyper, delta=1.):
    nu, logc = order_factors(order, data, delta)
    
    chi2_total = 0.
    logl = 0.
    
    for (t, d) in zip(*data):
        G = eval_G(t, x, order)
        
        b_hat, chi2, rank, s = np.linalg.lstsq(G, d, rcond=None)
        logdet = 2.*sum(np.log(s))
        b_regularizer = np.dot(b_hat, b_hat)/(delta**2)
        
        chi2_total += float(chi2)
        logl += -logdet/2 - b_regularizer/2
    
    logl += logc - nu/2*np.log(chi2_total)
    return logl