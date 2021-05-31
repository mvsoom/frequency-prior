% Obtaining complete posterior samples


This document details how to obtain complete posterior samples
$(b,\theta,\sigma)$ given the nested sampling samples and weights of $\theta$.

We use notation and equations from @VanSoom2020 but vectors are not in
boldface. $D$ stands for all data $d_{1:n}$. We use the shorthand $b$ for all
$b_{1:n}$ (which is a set of vectors $b_i$). 

# Decomposition of the posterior

Denote the acquired nested sampling samples as $\theta_{1:J}$ with estimated
weights $w_{1:J}^{(\theta)}$.

To get a *complete* posterior sample $(b,\theta,\sigma)$ from
$p(b,\theta,\sigma|D)$ for a sample $\theta \in \theta_{1:J}$ with weight
$w^{(\theta)}$, we first sample $\sigma$ from $p(\sigma|\theta,D)$ and then
sample $b_i$ from $p(b_i|\sigma,\theta,D)$. This is based on the decomposition

$$ p(b,\theta,\sigma|D) = p(\theta|D) \times p(\sigma|\theta,D) \times
p(b|\theta,\sigma,D). $$

This decomposition also tells us the weight $w^{(b,\theta,\sigma)}$ to assign
to the newly acquired sample $(b,\theta,\sigma)$:

$$ w^{(b,\theta,\sigma)} = w^{(\theta)} \times p(\sigma|\theta,D) \times
p(b|\theta,\sigma,D). $$

Note: though nested sampling can estimate the actual posterior density at a
given point (because of the knowledge of $Z$), in practice we renormalize the
importance weights as we use them to calculate weighted statistics. Thus in the
code the log weights are defined up to addition of a constant.

# Sampling $\sigma$

$$ p(\sigma|\theta,D) = \frac{p(\theta,\sigma|D)}{p(\theta|D)} \propto p(\sigma) p(D|\theta,\sigma) $$

Given that

$$ Z(P,Q) \equiv \int d\theta p(\theta) \int d\sigma p(\sigma) p(D|\theta,\sigma), $$

the expression for $p(D|\theta,\sigma)$ can be gathered from Eq. (19) by
identification. This yields

$$ p(\sigma|\theta,D) \propto \sigma^{-1} (\sigma^2)^{\frac{nm-N}{2}}
\exp{-\sum_i \chi_i^2/2\sigma^2}. $$

This can be transformed into an inverse gamma distribution by the
transformation

$$ z = \sigma^2 $$

such that (using [Wikipedia's convention][1])

$$ z \sim \text{Inv-Gamma}(\alpha,\beta) $$

where $\alpha = (N-nm)/2$ and $\beta = \sum_i \chi_i^2/2$.

[1]: https://en.wikipedia.org/wiki/Inverse-gamma_distribution

# Sampling $b_i$

Let $b = b_{1:n}$.

$$ p(b|\theta,\sigma,D) = \frac{p(b,\theta,\sigma|D)}{p(\theta,\sigma|D)}
\propto p(b,\theta,\sigma|D)$$

This is just the posterior, so from Eq. (16),

$$ p(b|\theta,\sigma,D) \propto \prod_i \exp{-Q_F^{(i)}/2} $$

Using the approximation in Eq. (18), we have finally

$$ p(b_i|\theta,\sigma,D) \propto \exp{-\frac{1}{2\sigma^2} (b_i - \hat{b_i})^T g_i (b_i - \hat{b_i})} $$

or

$$ b_i \sim \text{Multivariate-Normal}(\hat{b_i}, \sigma^2 g_i^{-1}) $$ {#eq:b}

Note that because of the approximation (18) this distribution is independent of
the prior $p(b)$ used; it only plays a role in the likelihood function.[^a] In
addition, note that we do not need to calculate $g_i^{-1}$ to sample from
@eq:b: see for example [here](https://stackoverflow.com/a/16708868/6783015).

[^a]: Come to think of it, couldn't we integrate out $\delta$ in the same way
as $\sigma$?
