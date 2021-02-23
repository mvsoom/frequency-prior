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

# # Results from running Bretthorst's program
#
# ----
#
# **Note:** we use the Peterson and Barney (1952) data for *male* speakers only.
#
# ----
#
# We find the Lagrange multipliers describing the distributions for the decoupled $u_j$ variables using Bretthorst's program and report the results here. We also write out the `uXXX_icdf.grid` files required for sampling from these informative priors.
#
# > Although Bretthorst's program cannot deal with an invariant measure, because ours is linear in the variables it does not make a difference -- we simply need to subtract constants from the inferred $\lambda$ values when converted to ordinary poly basis (see below).
#
# **Update:** On second thought, this might be incorrect, since the expression is $\sum p_k \log p_k/m_k \rightarrow \sum p_k \log q_k$ which is not in the correct form. So we might have to use the `maxentropy` library after all, with:
# - Legendre polynomials as moment functions instead of ordinary polynomials
# - Fixed value of $J$: we already know this from these results, which we can view as an approximation (i.e. without invariant measure) to the real results.
#
# ## Note on implementation
#
# Bretthorst used Legendre polynomials in the ME pdf rather than the ordinary $(x, x^2, x^3, \cdots)$ basis. This is a good move because the $\lambda$ values are more like $N(0,1)$ in this basis.
#
# This means that we have to use the Legendre basis to calculate the ME pdf, as is done in `do()` below. We can convert to ordinary polynomial basis $(x, x^2, x^3, \cdots)$ easily.
#
# Note that the invariant measure is given in the ordinary basis, e. g. $\exp\{2s + t\}$, so we need to convert to this basis to give the proper results.
#
# ## Note on the ME pdf domain
#
# Bretthorst uses a discretized grid for his calculations. This has the consequence that the inferred ME pdfs may not integrate to a finite number when considered over the domain $u \in [0,\infty)$. In fact this happens for $u_2$ and $u_3$ below. Therefore we limit the domain to the min and max values of the observed samples (as is done in the software). This is consistent with setting the bounds for the Jeffreys priors to the min and max observed values.

# !python3 data.py

# +
# %pylab inline
import data
import gvar

def get_lambdas(report, m):
    table = report.splitlines()[-m:]
    
    def parse_lambda(line):
        cols = line.split()
        x, sd = cols[3], cols[4]
        return gvar.gvar(x, sd)
    
    lambdas = np.array([
        parse_lambda(line) for line in table
    ])
    return lambdas

def do(name, m, u_max=None, n=1000, num_samples=10000, lambdas=None):
    from IPython.display import display
    
    u = eval("data." + name) # E.g. data.u1
    if u_max is None:
        u_max = u.max()
    
    # Get Lagrange lambdas based on the Legendre polys
    if lambdas is None:
        file = f'{name}/BayesOtherAnalysis/DensityEstimationMaxEnt.mcmc.values'
        with open(file, 'r') as f: report = f.read()
        lambdas = get_lambdas(report, m)

    print(f'lambdas[1:{m}] [using Legendre polynomials] =', lambdas)
    
    # Calculate the unnormalized pdf `q` and recalculate Z using a simple
    # discrete sum (we ignore the software's logZ estimate)
    coeff = [0, *gvar.mean(lambdas)]
    domain = (u.min(), u.max())
    legendre = np.polynomial.Legendre(coeff, domain=domain)
    
    x, dx = np.linspace(0, u_max, n, retstep=True)
    q = np.exp(legendre(x))
    qcdf = np.cumsum(q)*dx
    Z = qcdf[-1]
    
    p = q/Z
    pcdf = qcdf/Z
    pcdf[0] = 0.
    
    # Write out inverse cdf points
    icdf_grid = np.vstack([pcdf, x]).T
    np.savetxt(f'{name}_icdf.grid', icdf_grid)
    
    # Calculate inverse cdf for sampling
    icdf = lambda v: np.interp(v, pcdf, x)
    
    samples = icdf(np.random.rand(num_samples))
    
    # Plot pdf, histogram of data, and samples from ME pdf
    figsize(12,4)
    subplot(121); title('Data vs. ME pdf')
    hist(u, density=True)
    plot(x, p)
    subplot(122); title('Samples from ME pdf')
    hist(samples, bins=50, density=True)
    plot(x, p)
    
    # Transform Lagrange lambdas based on Legendre polys to ordinary poly.
    # The lowest-order (constant) term fixes the normalization and can be
    # ignored.
    poly = np.polynomial.Polynomial.cast(legendre)
    print(f'lambdas[1:{m}] [using ordinary polynomials] =', poly.coef[1:])
    
    print('Underlying Legendre polynomial =')
    display(legendre)
    print('Underlying ordinary polynomial =')
    display(poly)


# -

# ## `u1` $(K = 4)$

# !cat u1/BayesOtherAnalysis/DensityEstimationMaxEnt.mcmc.values

do("u1", 4)

# ## `u2` $(K=7)$

# !cat u2/BayesOtherAnalysis/DensityEstimationMaxEnt.mcmc.values

do("u2", 7)

# ## `u3` $(K=5)$

# !cat u3/BayesOtherAnalysis/DensityEstimationMaxEnt.mcmc.values

# +
report = """
Lagrange Multiplier 1              -2.30512E+00    1.88883E-01   -2.30293E+00
Lagrange Multiplier 2              -1.48841E+00    1.89786E-01   -1.44317E+00
Lagrange Multiplier 3               1.47433E-01    1.71039E-01    1.90514E-01
Lagrange Multiplier 4               8.84790E-01    2.09496E-01    8.98417E-01
Lagrange Multiplier 5               2.42291E+00    2.14912E-01    2.40688E+00
"""

lambdas = get_lambdas(report, 5)

do("u3", 5, lambdas=lambdas)
# -


