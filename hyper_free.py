import numpy as np
from aux import product
import arctic

P_max = 10
Q_max = 10
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

    new = (True,)
    Ps = range(0, P_max+1)
    Qs = range(1, Q_max+1)

    # P varies fastest, then Q, then `new`
    return list(product(new, Ps, Qs, order=(0, 2,1)))

def get_hyperparameters():
    # Note that the model clips all frequencies >= fs/2, so we do not
    # have to take that into account in the frequency bounds.
    global Q_max

    bounds = {
        'b': [(10.,)*Q_max, (500.,)*Q_max], # All bandwidths in [20, 500]
        'f': None
    }

    x0 = 150
    Ex = 500.*np.arange(1, Q_max+1)  # Schwa model
    F = [x0, *Ex]
    
    delta = 1.
    
    return bounds, F, delta
