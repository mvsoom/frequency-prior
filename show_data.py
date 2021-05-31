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

# +
# %pylab inline

import hyper

def show(file):
    fs, ts, ds = hyper.get_data(file, 11000)
    d = hstack(ds)
    t = arange(len(d))/fs # sec
    plot(t, d)
    xlabel("time [sec]")


# -

show('awb/arctic_a0094')

show('bdl/arctic_a0017')

show('jmk/arctic_a0067')

show('rms/arctic_a0382')

show('slt/arctic_b0041')
