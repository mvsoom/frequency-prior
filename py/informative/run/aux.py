import numpy as np
from datetime import datetime
import band_bounds as bandwidths
import freq_bounds as frequencies
from product import product
import arctic

def get_data(file, resample):
    fs0, fs, ts, ds, _ = arctic.load_arctic_file(
        f'arctic/{file}.wav', f'arctic/{file}.marks', resample
    )

    ts = [t/fs for t in ts] # Give units [sec]

    data = (fs, ts, ds)
    return data

def get_grid(P_max, Q_max):
    new = (False, True)
    Ps = range(0, P_max+1)
    Qs = range(1, Q_max+1)

    # P varies fastest, then Q, then `new`
    return list(product(new, Ps, Qs, order=(0, 2,1)))

def get_hyperparameters():
    # Note that the model clips all frequencies >= fs/2, so we do not
    # have to take that into account in the frequency bounds.
    bounds = {
        'b': bandwidths.bounds(),
        'f': frequencies.bounds()
    }

    x0 = bounds['f'][0][0] # Lower bound of F1
    F = [x0]
    
    delta = 1.
    
    return bounds, F, delta

def log(*args, **kwargs):
    time = datetime.now().strftime("%d%b%y %H:%M:%S")
    print(f'{time}', *[f'{k}={v}' for k,v in kwargs.items()], *args, sep='\t')