import sys
import csv
import model
import driver_script
from hyper import get_grid, get_data, get_hyperparameters
from aux import log

output_file = "post/run_stats.csv"

fieldnames = [
    'file',
    'runid',
    'new',
    'P',
    'Q',
    'nlive',
    'niter',
    'logl',
    'logz',
    'logzerr',
    'information',
    'walltime'
]

def make_csv(output_file):
    f = open(output_file, 'w')
    writer = csv.DictWriter(f, fieldnames)
    writer.writeheader()
    return writer

def write_stats(writer, input_file, new, P, Q, results):
    r = results
    writer.writerow({
        'file': input_file,
        'runid': driver_script.runid,
        'new': new,
        'P': P,
        'Q': Q,
        'nlive': r.nlive,
        'niter': r.niter,
        'logl': r.logl[-1],
        'logz': r.logz[-1],
        'logzerr': r.logzerr[-1],
        'information': r.information[-1],
        'walltime': r.walltime
    })

if __name__ == "__main__":
    writer = make_csv(output_file)

    for input_file in sys.argv[1:]:
        log("Processing", input_file=input_file, output_file=output_file)
        
        grid = get_grid(10, 5)
        data = get_data(input_file, 11000)
        hyper = get_hyperparameters()
        
        for (new, P, Q) in grid:
            results = driver_script.run_nested(new, P, Q, data, hyper)
            write_stats(writer, input_file, new, P, Q, results)