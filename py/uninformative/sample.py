# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Sample from $p(\boldsymbol x|\boldsymbol F)$

# %pylab inline
import scipy.stats


# +
def sample_x(J, x0, Ex, size=1, u=None):
    # Concatenate hyperparameters
    F = [x0, *Ex]
    
    # Calculate scale parameters for the u ~ Exp(beta) such
    # that the marginal moments agree with Ex
    beta = [(F[j+1] - F[j])/F[j+1] for j in range(J)]
    
    # Draw the u
    u = scipy.stats.expon.rvs(scale=beta, size=(size,J))
    
    # Transform to x
    x = vstack([x0*exp(sum(u[:,0:j+1], axis=1)) for j in range(J)])
    
    return x if size > 1 else x[:,0] # (J, size)

x0 = 200
Ex = (500, 1000, 1500, 2000)

samples = sample_x(4, x0, Ex, size=1000000)
mean(samples, axis=1)
# -

U = 3000
B = 50
for x in samples:
    hist(x[x<U], bins=B)
    axvline(mean(x))
    show()

# ## PPF version for use in nested sampling

# +
from sample_ppf import sample_x_ppf

J = 4
x0 = 450
Ex = (800, 2000, 2500, 4000)

samples = vstack([sample_x_ppf(J, x0, Ex, q = rand(J)) for _ in range(10000)])
mean(samples,axis=0)
