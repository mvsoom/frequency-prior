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

# Notes:
#
# - https://github.com/PythonCharmers/maxentropy/
#
# - Don't forget to add invariant measure $m(\boldsymbol u)$ for $u_j$, $j > 1$. **Update:** this is probably not necessary. Since the $m(\boldsymbol u)$ is separable and just induces a shift in the Lagrange multiplier of the first moment constraint, we can just solve the problem assuming $m(u_j) \propto 1$, yielding a $\alpha_j^{(1)}$, and transform this to $\lambda_j^{(1)} = \alpha_j^{(1)} - c$ for the original problem with $m(u_j) \propto \exp(c u_j)$.
#
# - This package can deal with correlated features $F(\boldsymbol u)$. Actually it can handle 100s of parameters or more with high dimensionality of $\boldsymbol u$. Currently I choose dense matrix formant and the Powell method for optimization, which is robust in small problems.

# +
# %pylab inline

import scipy.stats
import maxentropy
import pandas

def support(z):
    return linspace(min(z), max(z), 1000)

data = pandas.read_csv("../../data/pb.csv")

# Don't forget to add invariant measure m(u) for other $u_j$!
# Called `prior_log_pdf` in next cell.
x = array(data['F1'])
x0 = amin(x)
u = log(x/x0)


# +
def F(k):
    def F(u):
        return u**k
    return F

K = 4

features = [F(k+1) for k in range(K)]
target_expectations = array([mean(F(u)) for F in features])

def make_sampler(x0, F1, n=10**5):
    # Use our uninformative prior as importance sampler
    beta = (F1 - x0)/(F1)
    Exp = scipy.stats.expon(scale=beta)
    
    def sampler():
        u = Exp.rvs(size=(n,1))
        logp = Exp.logpdf(u).sum(axis=1)
        return u, logp
    
    return sampler

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


