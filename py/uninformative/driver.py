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
# %run driver.ipy

import arctic

import joblib
memory = joblib.Memory('cache', verbose=1)

# +
marks = """
Time_s
0.521712220149
0.529017772074
0.536285752548
0.543491841004
"""

fs0, fs, ts, ds, _ = arctic.load_arctic_file('bdl/arctic_a0017.wav', marks, 8000)

ts = [t/fs for t in ts] # Give units [sec]

data = (ts, ds)

# +
# (P, Q) grid
Ps = np.arange(0,11)
Qs = (1,2,3,4)

# Hyperparameters
bounds = {
    'α': [(40., 40., 60., 60.), (180., 250., 420., 420.)],
    'ω': [(300., 1000., 2000., 2500.), (900., 2000., 3000., 4000.)]
}

x0 = 300.
Ex = array([600.,1700.,2500.,3800.]) # Roughly base on Vallee (1994)
F = [x0, *Ex]
# -


