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