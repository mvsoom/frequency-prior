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

# # Informative case: moment density estimation on decoupled $(u_j$
#
# Conclusion: works well. $K = 2$ moments $(<u_1>,<u_1^2>)$ suffice to capture the marginal $p(u_1|I_K)$. $K = 4$ performs reasonably well for $u_2$. $u_3$ is harder, $K = 12$ is needed to get the bimodal behavior and is the maximum value of $K$ that will return meaningfull distributions.
#
# Notes:
#
# - $J$ is determined by our dataset; in the Peterson (1952) case, this is $J=3$.
#
# - https://github.com/PythonCharmers/maxentropy/
#
# - Don't forget to add invariant measure $m(\boldsymbol u)$ for $u_j$, $j > 1$. **Update:** this is very likely not necessary (but still need to double check). Since the $m(\boldsymbol u)$ is separable and just induces a shift in the Lagrange multiplier of the first moment constraint, we can just solve the problem assuming $m(u_j) \propto 1$, yielding a $\alpha_j^{(1)}$, and transform this to $\lambda_j^{(1)} = \alpha_j^{(1)} - c$ for the original problem with $m(u_j) \propto \exp(c u_j)$.
#
# - This package can deal with correlated features $F(\boldsymbol u)$. Actually it can handle 100s of parameters or more with high dimensionality of $\boldsymbol u$. Currently I choose dense matrix formant and the Powell method for optimization, which is robust in small problems.
#
# - Sampling from these distributions can be done with 1D slice sampling, or, even simpler, 1D rejection sampling.

# +
# %pylab inline

import scipy.stats
import maxentropy
import pandas

data = pandas.read_csv("../../data/pb.csv")

x = array(data['F1'])
x0 = amin(x)
y = array(data['F2'])
z = array(data['F3'])


# +
def support(z):
    return linspace(min(z), max(z), 1000)

def F(k):
    def F(u):
        return u**k
    return F

def make_sampler(a, b, n=10**5):
    # Use our uninformative prior as importance sampler
    beta = (b - a)/b
    Exp = scipy.stats.expon(scale=beta)
    
    def sampler():
        u = Exp.rvs(size=(n,1))
        logp = Exp.logpdf(u).sum(axis=1)
        return u, logp
    
    return sampler


# -

# ## Expand $u_1$

# +
u = log(x/x0)
K = 2

features = [F(k+1) for k in range(K)]
target_expectations = array([mean(F(u)) for F in features])

sampler = make_sampler(x0, mean(x), n=10**5)

model = maxentropy.skmaxent.MCMinDivergenceModel(
    features, sampler, prior_log_pdf=None, vectorized=True,
    matrix_format='ndarray', algorithm='CG', verbose=0
)
# -

model.fit(target_expectations[None,:])
model.params

model.expectations()

target_expectations

# +
hist(u, bins=30, density=True)

pdf = model.pdf(model.features(support(u)))
plot(support(u), pdf)
# -
# ## Expand $u_2$


# +
u = log(y/x)
K = 8

features = [F(k+1) for k in range(K)]
target_expectations = array([mean(F(u)) for F in features])

sampler = make_sampler(mean(x), mean(y), n=10**5)

model = maxentropy.skmaxent.MCMinDivergenceModel(
    features, sampler, prior_log_pdf=None, vectorized=True,
    matrix_format='ndarray', algorithm='CG', verbose=0
)

model.fit(target_expectations[None,:])

print('Lagrange multipliers:', model.params)
print('Model expectations', model.expectations())
print('Target expectations', target_expectations)

hist(u, bins=30, density=True)

pdf = model.pdf(model.features(support(u)))
plot(support(u), pdf)
# -

# ## Expand $u_3$
#
# $K = 11,12$ is needed to get the bimodal behavior.
#
# $K > 12$ leads to a uniform dsitrubution-like curve with the minimizer `'CG'`.

# +
u = log(z/y)
K = 11

features = [F(k+1) for k in range(K)]
target_expectations = array([mean(F(u)) for F in features])

sampler = make_sampler(mean(y), mean(z), n=10**5)

model = maxentropy.skmaxent.MCMinDivergenceModel(
    features, sampler, prior_log_pdf=None, vectorized=True,
    matrix_format='ndarray', algorithm='CG', verbose=0
)

model.fit(target_expectations[None,:])

print('Lagrange multipliers:', model.params)
print('Model expectations', model.expectations())
print('Target expectations', target_expectations)

hist(u, bins=30, density=True)

pdf = model.pdf(model.features(support(u)))
plot(support(u), pdf)
