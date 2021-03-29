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

# # Testing the samplers

# ## Sample from $p(\boldsymbol x|\boldsymbol F)$

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

num = 1000000
J = 3

x0 = 200
Ex = 500*(1 + arange(J))
print(Ex)

samples = sample_x(J, x0, Ex, size=num)
mean(samples, axis=1)
# -

# The result of imposing an upper bound on the `x` is a relatively large shift in the means away from the given values (constraints), despite the high acceptance ratio. The heavy tails of the pdfs for the frequency ratios manifest themselves.

fs = 11000
keep = all(samples < fs/2, axis=0)
print(f"Acceptance rate at fs = {fs}:", sum(keep)/num)
accept = samples[:,keep]
mean(accept, axis=1)

U = 3000
B = 50
for x in samples:
    hist(x[x<U], bins=B)
    axvline(mean(x))
    show()

# ### PPF version for use in nested sampling

# +
from sample_ppf import sample_x_ppf

J = 4
x0 = 450
Ex = (800, 2000, 2500, 4000)

F = [x0, *Ex]

samples = vstack([sample_x_ppf(rand(J), J, F) for _ in range(10000)])
mean(samples,axis=0)
# -

# ## Testing PPF Jeffreys sampler

# +
from sample_ppf import sample_jeffreys_ppf

bounds = [(300., 1000., 2000., 2500.), (900., 2000., 3000., 4000.)]
J = 3

samples = vstack([sample_jeffreys_ppf(rand(J), bounds) for _ in range(100000)])
for x in samples.T:
    hist(x, bins=B)
    axvline(mean(x))
    show()
# -

# ## Testing `ptform_XXX()`

# +
import model
import hyper_cmp

hyper = hyper_cmp.get_hyperparameters()
print(hyper)
# -

# ### `ptform_old`

# +
Q = 2
order = (None, Q)

samples = vstack([model.ptform_old(rand(Q*2), order, hyper) for _ in range(100000)])
for x in samples.T:
    hist(x, bins=B)
    axvline(mean(x))
    show()
# -

# ### `ptform_new`

# +
Q = 4
order = (None, Q)

samples = vstack([model.ptform_new(rand(Q*2), order, hyper) for _ in range(100000)])

U = 6000
B = 50
for x in samples.T:
    hist(x[x<U], bins=B)
    axvline(mean(x))
    show()
