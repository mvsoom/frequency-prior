import numpy as np
import scipy.stats

def sample_jeffreys_ppf(q, bounds):
    J = len(q)
    lower, upper = bounds
    
    assert len(lower) >= J and len(upper) >= J
    
    a = np.log(lower[:J])
    b = np.log(upper[:J])
    x = a + q*(b-a)
    return np.exp(x)

def sample_x_ppf(q, J, F):
    assert J == len(q) and len(F) >= J + 1

    # Calculate scale parameters for the u ~ Exp(beta) such
    # that the marginal moments agree with Ex
    beta = [(F[j+1] - F[j])/F[j+1] for j in range(J)]
    
    # Draw the u
    u = np.atleast_1d(scipy.stats.expon.ppf(q, scale=beta))
    
    # Transform to x
    x0 = F[0]
    x = [x0*np.exp(np.sum(u[0:j+1])) for j in range(J)]
    return np.array(x)