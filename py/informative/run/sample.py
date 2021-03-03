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
import sample_ppf
import gvar

figsize(12, 3)
for i in [1, 2, 3]:
    subplot(1, 3, i)
    samples = sample_ppf.ppf[i-1](rand(1000000))
    title(f'Samples from ME pdf for $u_{i}$')
    hist(samples, bins=50)
    xlabel(f'$u_{i}$')

figsize(12, 3)
for i in [1, 2, 3]:
    subplot(1, 3, i)
    samples = sample_ppf.ppf[i-1](rand(1000000))
    title(f'Samples from ME pdf for $u_{i}$')
    hist(exp(samples), bins=50)
    xlabel(f'$\exp(u_{i})$')

# +
J = 3
F = [190]
n = 10000

samples = np.array([sample_ppf.sample_x_ppf(rand(J), J, F) for _ in range(n)])

accept = np.all(samples < 5500, axis=1)
print("Acceptance ratio:", sum(accept)/n)

samples = samples[accept]

hist(samples, bins=20);

# +
x, y, z = samples.T

mean = np.mean(samples, axis=0)
cov = np.cov(samples, rowvar=False)
gvar.gvar(mean, cov)
# -

figsize(8,6)
subplot(311)
hist(x, bins=20)
xlim(190, 5500)
subplot(312)
hist(y, bins=50)
xlim(190, 5500)
subplot(313)
hist(z, bins=50)
xlim(190, 5500)

hist2d(x, y, bins=40);

hist2d(y, z, bins=50);


