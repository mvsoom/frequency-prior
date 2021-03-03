import numpy as np
from datetime import datetime
import gvar

def log(*args, **kwargs):
    time = datetime.now().strftime("%d%b%y %H:%M:%S")
    print(f'{time}', *[f'{k}={v}' for k,v in kwargs.items()], *args, sep='\t')

def sample_gvar(g, size=None):
    """Fast sampling of gvar arrays"""
    mean = gvar.mean(g)
    
    cov = np.zeros((len(g.flat), len(g.flat)), float)
    for idx, bcov in gvar.evalcov_blocks(g):
        cov[idx[:, None], idx] = bcov

    samples = np.random.multivariate_normal(mean, cov, size=size)
    return samples

# https://stackoverflow.com/a/9969179/6783015
import itertools

def product(*args, order=None):
    """itertools.product() with custom generation order"""
    if order is None:
        order = range(len(args))

    prod_trans = tuple(zip(*itertools.product(
        *(args[axis] for axis in order))
    ))

    prod_trans_ordered = [None] * len(order)
    for i, axis in enumerate(order):
        prod_trans_ordered[axis] = prod_trans[i]

    return zip(*prod_trans_ordered)