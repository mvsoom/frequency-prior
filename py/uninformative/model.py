import numpy as np
from sample_ppf import sample_jeffreys_ppf, sample_x_ppf

import joblib
memory = joblib.Memory('cache', verbose=1)

#@memory.cache
def run_nested(order, data, hyper):
    P, Q = order
    print(order)
    return (P, Q)

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