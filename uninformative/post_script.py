import sys
import csv
import model
import driver_script
import importlib
from aux import log

output_file = "post/run_stats.csv"

fieldnames = [
    'hyperfile',
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

def splitargs(args):
    hyper_files = [a for a in args if 'hyper' in a]
    input_files = [a for a in args if not a in hyper_files]
    return hyper_files, input_files

def make_csv(output_file):
    f = open(output_file, 'w')
    writer = csv.DictWriter(f, fieldnames)
    writer.writeheader()
    return writer

def write_stats(writer, hyper_file, input_file, new, P, Q, results):
    r = results
    writer.writerow({
        'hyperfile': hyper_file,
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

    hyper_files, input_files = splitargs(sys.argv[1:])

    for hyper_file in hyper_files:
        hyper_module = importlib.import_module(hyper_file)
        for input_file in input_files:
            log(
                "Processing",
                hyper_file=hyper_file,
                input_file=input_file,
                output_file=output_file
            )

            grid = hyper_module.get_grid()
            data = hyper_module.get_data(input_file)
            hyper = hyper_module.get_hyperparameters()

            for (new, P, Q) in grid:
                results = driver_script.run_nested(new, P, Q, data, hyper)
                write_stats(writer, hyper_file, input_file, new, P, Q, results)