import numpy as np
import scipy.interpolate

_icdf_grid = [np.loadtxt(f'../bretthorst/u{i}_icdf.grid', unpack=True) for i in [1, 2, 3]]
ppf = [scipy.interpolate.interp1d(cdf, u) for cdf, u in _icdf_grid]

def sample_jeffreys_ppf(q, bounds):
    J = len(q)
    lower, upper = bounds
    
    assert len(lower) >= J and len(upper) >= J
    
    a = np.log(lower[:J])
    b = np.log(upper[:J])
    x = a + q*(b-a)
    return np.exp(x)

def sample_x_ppf(q, J, F):
    assert J == len(q) and len(F) == 1
    
    x = np.empty(J)
    x[-1] = F[0] # == x0
    
    for j in range(J):
        u = ppf[j](q[j])
        x[j] = np.exp(u)*x[j-1]

    return x