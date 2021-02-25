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