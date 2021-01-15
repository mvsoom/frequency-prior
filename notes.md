% Prior for Bayesian formant tracking


TODO: Check follow-up paper of this one: @Turner2009

REMINDER: The evidence of a hypothesis is the most important quantity! Don't
mesmerize, try and compute!

AIM: A "sophisticatedly simple" prior for formant frequencies of vowels and
consonants for use in Bayesian formant tracking.

We derive a prior for formant frequencies based on the observation of
@Turner2003, to be used for **Bayesian formant tracking**. The prior represents
a good balance between complexity and accuracy. We show this by performing
model comparison for our prior and other common priors on classic datasets such
as @Peterson1952.

Note: "Bayesian formant tracking" is the term used in the literature.

# Discussion of @Turner2003

## Dimensionality reduction

It is often said that the 2D subspace (F1,F2) contains enough information for
vowel classification. But in this paper it is suggested that there is an even
better subspace $(\theta,\phi)$: of equal dimension but which makes use of F3
information. For further support of the superiority of this claim one should
compare vowel classification in (F1,F2) vs $(\theta,\phi)$.

Observe that the $(\theta,\phi)$ coordinates are dimensionless and invariant
under rescaling the **formant vector** $(x,y,z)$, being function of $y/x$ and
$z/r$ respectively. How can we translate this to a prior?

## Formant frequency clusters

Fig. 1

- Note that while these clusters are relatively well separated at the $1\sigma$
  level, these encompass only 30% of the data points. The implied data density
  (i.e. a mixture of Gaussians defined by these clusters) is thus a series of
  broad peaks which are close to each other. Recognition of vowels based only
  on F1/F2 without any other information thus remains ambiguous.

- The ellipsoids are not axis-aligned: there are significant positive
  ("radially outward") correlations.

## Transformations and basis dependence

We note in passing that the success rate $q$ of vowel classification, which is
based on "leave-one-out" ML estimates for a mixture of Gaussians (i.e. K
nearest neighbours), depends on the representation of the space:

- $(x,y,z)$: $q = 0.84$

- $(r,\theta,\phi)$: $q = 0.86$

This difference is an example of what MacKay called "the opportunity to hunt
for a good basis" [@MacKay2005, p. 342], and shows that during inference we
have made an approximation (such as ML) or applied decision theory (such as
choosing a quadratic loss function) in such a way that made our result basis
dependent. In other words, applying a non-linear (but reversible)
transformation to the data can indeed change our answers if we stop
representing them as probability distributions, and shows the practical value
of feature engineering and the nonlinear transformations of neural nets.

## Transformation groups

TODO: See @Jaynes1968 and (maybe) @Jaynes1989a, @Jaynes1973, and [@Jaynes2003,
Ch. 12]

Which prior information $I$ do we choose to derive our prior $p(x,y,z)$ from?
Remember, $I$ is (just) an approximation to our the information we really have
about the problem; we may choose $I$ in a way that is useful to us.

- Assume variance of formants is roughly independent of vocal tract
  length,[^re] then $p(x,y,z)$ should be invariant under rescaling of the
  formant vector $(x,y,z)$ (and we automatically have independence of units
  used). Is this equivalent to the spherical cones argument on p. 3 and Fig. 3?

[^re]: Another way of phrasing this: vocal tract length is the biggest
contributor to (x,y,z) variance next to vowel class (as shown in this paper);
therefore, approximate by saying it is the only contributor.

- In 2D, use
  
  $$ p(x,y) = f(r) g(y/x) $$

  or

  $$ p(x,y) = f(r) h(\arccos(y/x)), $$

  where $\arccos(y/x) = \theta$?

# Formants, poles, predictor coefficients

*NOTE:* Read a paper about Bayesian formant tracking first (such as @Mark2003
or a more recent one): how do they separate the formants from the poles? These
thoughts were written before I have read such a paper.

Can the prior be used for poles and predictor coefficients?

Formants are a subset of the (sometimes many) poles used in AR modeling.
Although poles probably undergo the same scaling behavior as the formants, the
prior's hyperparameters $\theta$ which are to be determined from formant
measurements such as @Peterson1952 will not apply to poles in general, for
example poles used for "spectral balance" which do not represent true
resonances of the VT.

Predictor coefficients are simply expansion coefficients $a$ in the denominator
of the expansion of the VT transfer function (see [AR theory] below). Therefore
it is not clear whether the scaling behavior of the roots of that polynomial
(i.e. the poles) can be used to determine the prior $p(a)$. In [@Yoshii2013,
paragraph below Eq. (17)], $p(a) = N(0, \lambda I)$ where $\lambda$ is a
hyperparameter.

We conclude that our prior is "high-level" and does not apply easily to the
"lower-level" concepts of poles and predictor coefficients. The hyperparameters
$\theta$ are determined from tables containing high-level information, i.e. the
formants which are found either manually or by indirect methods such as peak
picking.

## AR theory

A $P$th order AR process is [@Yoshii2013]

$$ x_m = \sum_{p=1}^P a_p x_{m-p} + \epsilon_m $$

where the $a$ are a set of $P$ predictor coefficients and the $\epsilon$
represent the noise term or source signal. The linear AR system represents an
all-pole transfer function given by

$$ A(z) = 1/(1 - a_1 z^{-1} - \ldots - a_P z^{-P}) $$

such that

$$ X(z) = E(z) A(z) $$

where $X$ and $E$ are the transforms of $x$ and $\epsilon$.

# Extending to $n$ dimensions

The case is straightforward for $n = 1,2,3$. What for $n > 3$? The answer is
[hyperspherical
coordinates](https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates). The
pdf still factorizes as

$$ p(r, \theta_1, \ldots) \propto f(r) g(z_1) \ldots g(z_{n-1}) $$

where the $z_i$ are invariant to rescaling. (For example, for $n=2$ $z_1 =
F_2/F_1$.)

# Why prefer simpler form over Gaussian mixture?

Why not fit a Gaussian mixture to @Peterson1952 or similar and use this as a
prior? Possible answers:

0. We want a prior for both consonants and vowels

1. Mixture has possibly too much parameters: model complexity

2. Data is for steady state vowels: Our prior possibly has a smaller inductive
   bias (compared to the Gaussian mixture) which might be handy for Bayesian
   formant tracking as it incorporates less of the assumed steady-state in
   the @Peterson1952 data. In other worsd, we use the same data to estimate the
   parameters for all priors, but some priors have a stronger inductive bias
   towards steady-stateness than others

3. @Peterson1952 is just one dataset, and has its own flaws (for example,
   biased towards F0 [@Kent2018, Sec. 6.3])

We look for a "sophicatedly simple" model [@Jaynes1985, p. 6].

# Applications

Where can this prior be used?

Answer 1: As the starting point for the construction of a more informative
prior either by means of maximum entropy or Bayes' theorem.

Answer 2: As a prior for Bayesian formant tracking papers. Other possibilities:

- @VanSoom2020 and the sflinear model

- Possibly @Mehta2012

- @Mark2003: "Prior distribution of formants is uniform". See also the
  follow-up paper and its citations:
  <https://ieeexplore.ieee.org/abstract/document/1326048/?casa_token=njoCd43KXJMAAAAA:FaadnuqfdrBDqDicFpKgBb8C9i4GYHKJuHAADJLzH_OD3T9AP3JOcX4jxuTMiF2R0yCtiG8m1yvl>

# Journals

https://www.springer.com/journal/11009
