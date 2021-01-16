# -*- coding: utf-8 -*-
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

# - The prior ranges for $\alpha$ and $\omega$ were chosen given Praat's estimates, so we choose the hyperparameters given that we know that it was `/ae/`. From Vallee (1994) we get: $\langle \boldsymbol x \rangle = (610, 1706, 2450, 3795)$. We then round these numbers. The lowerbound $x_0$ is chosen as the lower bound of $F_1$'s interval used in the paper.

# +
# %pylab inline

import joblib
import model

# +
# %run driver.ipy

PQ_grid = get_PQ_grid()
data = get_data()
hyper = get_hyperparameters()

# +
# Output to stdout is not printed in the notebook. If this is desired,
# change backend to "multiprocessing" and uncomment the decorator
# @memory.cache of model.run_nested()
options = dict(n_jobs=8, verbose=10, timeout=None, backend="multiprocessing")

with joblib.Parallel(**options) as parallel:
    out = parallel(
        joblib.delayed(model.run_nested)(order, data, hyper) for order in PQ_grid
    )


# +
def ptform_old(q, order, hyper):
    P, Q = order
    bounds, F = hyper
    
    qb = q[:Q]
    b = sample_jeffreys(qb, bounds['b'])
    
    qf = q[Q:]
    f = sample_jeffreys(qf, bounds['f'])
    
    x = zeros()
    
    

def loglike(x, data, hyper):
    """Calculate the integrated likelihood p(θ|dI)"""
    ssv = self.ssv
    χ2_total = 0.
    ans = self.c

    for (t, d) in zip(*data):
        G = ssv.eval_G(t, x) # (N,m) where N = len(t)
        g = ssv.eval_g(G)

        logdet_Λ = logdet(g)
        b_hat = solve(g, G.T @ d)
        e = d - G @ b_hat
        χ2 = dot(e, e)

        χ2_total += χ2
        ans += -0.5*logdet_Λ - 0.5*dot(b_hat, b_hat)/self.δb**2

    ans += -0.5*self.ν*χ2_total
    return ans


class ssv_model:
    def __init__(
        self, data, P, Q, θbounds, θscale=None, δb = 1.
    ):
        import ssv
        
        self.data = data
        self.n = len(data[0])
        self.N = sum([len(d) for d in data[1]])
        self.ssv = ssv.model_function(P, Q, θbounds, θscale)
        self.δb = δb
        
        ts, ds = data
        for d in ds:
            if max(abs(d)) > 1.:
                warnings.warn('Data is not normalized')
        
        # Precalculate c(P, Q), the factor independent from θ
        self.ν = self.N - self.n*self.ssv.m
        self.c = log(0.5) - 0.5*self.ν*log(pi) + loggamma(0.5*self.ν) \
               - 0.5*self.n*self.ssv.m*log(2.*pi*self.δb**2)
    
    def __str__(self):
        return f'{self.__class__.__name__}(n={self.n},ssv={str(self.ssv)})'
    
    def _likelihood_nested(self, x):
        """Calculate the integrated likelihood p(θ|dI)"""
        ssv = self.ssv
        χ2_total = 0.
        ans = self.c
        
        for (t, d) in zip(*self.data):
            G = ssv.eval_G(t, x) # (N,m) where N = len(t)
            g = ssv.eval_g(G)
            
            logdet_Λ = logdet(g)
            b_hat = solve(g, G.T @ d)
            e = d - G @ b_hat
            χ2 = dot(e, e)
            
            χ2_total += χ2
            ans += -0.5*logdet_Λ - 0.5*dot(b_hat, b_hat)/self.δb**2
        
        ans += -0.5*self.ν*χ2_total
        return ans
    
    def _ptform_nested(self, u):
        ssv = self.ssv
        x = u*ssv._xrange + ssv._xlower
        return x

    def run_nested(
        self,
        type='static',
        num_workers=8,
        save_to_self=False,
        samplerargs={},
        runargs={'save_bounds': False},
        **kwargs
    ):
        """Run the nested sampling algorithm
        
        Args:
            type (str): Use 'static' or 'dynamic' sampling
            num_workers (int): Number of workers in the pool. If None, CPU count is used.
                A total of about 4 or 5 workers seems to be quite efficient.
            save_to_self (bool): If True, the return value of this function will be saved to
                self.fitres
            samplerargs (dict): Passed on to the sampler. Note that these depend on the sampler type
            runargs (dict): Passed on to sampler.run_nested()
            **kwargs: Passed on to self.nest_res_to_fitres()
        
        Returns:
            fitres (util.dotdict): Canonical fit result with at least the following attributes:
            
                    nest_res (deep copy of nested run results)
                    model_inst (self)
                    log_evidence
                    samples
                    weights
                    θ_moments
                    θ_quantiles
                    B
                    σ
                
                where nest_res (an util.dotdict) is the transformed sampler.results
                with the following extra information about the nested sampling run:

                    labels
                    sampler_type
                    samplerargs
                    runargs
                    model_inst
                    duration
                    date
        """
        if not type in ['static', 'dynamic']:
            raise ValueError(type)
        
        Sampler = dynesty.NestedSampler if type == 'static' else dynesty.DynamicNestedSampler
        if num_workers == 1: # Avoid this pickle crap whenever possible
            sampler = Sampler(
                loglikelihood=self._likelihood_nested,
                prior_transform=self._ptform_nested,
                ndim=self.ssv.r,
                **samplerargs
            )
            with util.Timer() as timer:
                sampler.run_nested(**runargs)
        elif num_workers > 1:
            with pathos.multiprocessing.Pool(num_workers) as pool:
                pool.size = num_workers
                sampler = Sampler(
                    loglikelihood=self._likelihood_nested,
                    prior_transform=self._ptform_nested,
                    ndim=self.ssv.r,
                    pool=pool,
                    **samplerargs
                )
                with util.Timer() as timer:
                    sampler.run_nested(**runargs)
        else:
            raise ValueError(num_workers)

        nest_res = self.undo_nonlinear_transform(sampler.results)

        # Add plot labels and other general information
        nest_res.labels = self.ssv.gen_xlabels()
        nest_res.sampler_type = type
        nest_res.samplerargs = samplerargs
        nest_res.runargs = runargs
        nest_res.duration = timer.interval # sec
        nest_res.date = datetime.datetime.now()
        
        # Derive canonical fit_res form
        fitres = self.nest_res_to_fitres(nest_res, **kwargs)
        if save_to_self:
            self.fitres = fitres
        return fitres
