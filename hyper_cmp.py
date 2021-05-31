import numpy as np
import band_bounds as bandwidths
import freq_bounds as frequencies
from aux import product
import arctic

P_max = 10
Q_max = 5
resample_rate = 11000

def get_data(file):
    global resample_rate

    fs0, fs, ts, ds, _ = arctic.load_arctic_file(
        f'arctic/{file}.wav', f'arctic/{file}.marks', resample_rate
    )

    ts = [t/fs for t in ts] # Give units [sec]

    data = (fs, ts, ds)
    return data

def get_grid():
    global P_max, Q_max

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
    Ex = np.array([500., 1000., 1500., 2000., 2500.]) # Schwa model
    F = [x0, *Ex]
    
    delta = 1.
    
    return bounds, F, delta