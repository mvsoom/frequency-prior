import sys
import joblib
import model
from hyper import get_grid, get_data, get_hyperparameters
from aux import log

runid = 0 # Can be used to disambiguate multiple runs for same arguments

# See ./md/nested_sampling_options.md for the logic behind these settings
joblibargs = {'n_jobs': -1, 'verbose': 50, 'backend': "multiprocessing"}
samplerargs = {'nlive': 500, 'bound': 'multi', 'sample': 'rslice', 'bootstrap': 10}
runargs = {'save_bounds': False}

def run_nested(new, P, Q, data, hyper):
    global runid, samplerargs, runargs
    
    # Coerce to ints to avoid triggering recalculations
    order = (int(P), int(Q))
    return model.run_nested(
        new, order, data, hyper, runid, samplerargs, runargs
    )

def driver(new, P, Q, data, hyper):
    log("Started", new=new, P=P, Q=Q)

    res = run_nested(new, P, Q, data, hyper)

    walltime_min = res['walltime']/60.
    log(f"Finished in {walltime_min:.2f} min", new=new, P=P, Q=Q)

if __name__ == "__main__":
    input_file = sys.argv[1]
    log("Started", input_file=input_file)
    
    grid = get_grid(10, 5)
    data = get_data(input_file, 11000)
    hyper = get_hyperparameters()

    with joblib.Parallel(**joblibargs) as parallel:
        parallel(
            joblib.delayed(driver)(
                new, P, Q, data, hyper
            ) for (new, P, Q) in grid
        )

    log("Finished", input_file=input_file)