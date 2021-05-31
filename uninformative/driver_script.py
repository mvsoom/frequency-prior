import sys
import joblib
import model
import importlib
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
    hyper_file = sys.argv[2]
    hyper_module = importlib.import_module(hyper_file)
    
    log("Started", input_file=input_file, hyper_file=hyper_file)
    
    grid = hyper_module.get_grid()
    data = hyper_module.get_data(input_file)
    hyper = hyper_module.get_hyperparameters()

    with joblib.Parallel(**joblibargs) as parallel:
        parallel(
            joblib.delayed(driver)(
                new, P, Q, data, hyper
            ) for (new, P, Q) in grid
        )

    log("Finished", input_file=input_file, hyper_file=hyper_file)