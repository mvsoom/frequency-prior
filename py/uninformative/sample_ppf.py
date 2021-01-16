import numpy as np
import scipy.stats

def sample_x_ppf(J, x0, Ex, q):
    # Concatenate hyperparameters
    F = [x0, *Ex]
    
    # Calculate scale parameters for the u ~ Exp(beta) such
    # that the marginal moments agree with Ex
    beta = [(F[j+1] - F[j])/F[j+1] for j in range(J)]
    
    # Draw the u
    u = np.atleast_1d(scipy.stats.expon.ppf(q, scale=beta))
    
    # Transform to x
    x = [x0*np.exp(np.sum(u[0:j+1])) for j in range(J)]
    return np.array(x)