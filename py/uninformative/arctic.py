"""Most code is copy-pasted or adapted from sflinear/model.py"""
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize, getfigs
from scipy.io import wavfile
import scipy.signal
import parse

def parse_marks(marks):
    try:
        with open(marks, 'r') as f:
            marks_content = f.read()
        marks = marks_content # Parse downstream
    except:
        pass 
    
    res = []
    for line in marks.splitlines():
        line_res = parse.search('{:f}', line) # Extract first floating point number
        if line_res:
            res.append(line_res.fixed[0])
    return np.asarray(res)

def validate_marks(marks):
    if not (len(marks) > 1 and np.all(np.diff(marks) > 0.)):
        raise ValueError('Invalid marks')

def resample_data(fs, data, fs_new):
    N = data.shape[0]
    N_new = int(N*fs_new/fs)
    new_data = scipy.signal.resample(data, N_new, axis=0)
    return fs_new, new_data
    
def marks_to_array_indices(fs, d, marks, d_start_time=0.):
    praat_t = d_start_time + np.arange(len(d)) / fs
    indices = np.argmin(np.abs(praat_t[:,None] - marks[None,:]),axis=0)
    return indices

def ensure_stereo(data):
    try:
        x, y = data.T
        return x, y
    except ValueError as e1:
        try:
            x = data
            y = np.zeros(len(data))
            return x, y
        except Exception as e2:
            del e2
            raise ValueError('Cannot coerce to stereo') from e1

def extract_from_data(fs, data, begin_marker, end_marker):
    d_full, egg_full = ensure_stereo(data)
    begin_end = np.array([begin_marker, end_marker])
    split_indices = marks_to_array_indices(fs, d_full, begin_end)
    d = np.split(d_full, split_indices)[1]
    egg = np.split(egg_full, split_indices)[1]
    return d, egg

def extract_pitch_periods_from_data(fs, data, marks):
    d_full, egg_full = ensure_stereo(data)
    split_indices = marks_to_array_indices(fs, d_full, marks)
    ds = np.split(d_full, split_indices)[1:-1]
    eggs = np.split(egg_full, split_indices)[1:-1]
    return ds, eggs

def rescale_to_normalize_ds(ds, eggs):
    factor = 1./np.max([np.abs(d).max() for d in ds])
    return [d*factor for d in ds], [egg*factor for egg in eggs]

def apply_polarity(ds, eggs, polarity):
    def multiply(iter, c):
        return [c*x for x in iter]
    try:
        ds = multiply(ds, polarity[0])
        eggs = multiply(eggs, polarity[1])
    except TypeError:
        ds = multiply(ds, polarity)
        eggs = multiply(eggs, polarity)
    return ds, eggs

def load_arctic_file(path, marks, resample=False, polarity=+1):
    """
    Args:
        path (str): Path to wav file with speech signal in ch 1 and EGG in ch 2.
        marks (str): If str, the marks are given in seconds by the first float
            on each line in the string (if any). If a path to a text file,
            the contents of this file will be used as the string.
        polarity (float or 2-tuple): If tuple, polarity = (d_polarity, egg_polarity).
            Otherwise the same polarity is applie to both.
    
    Returns:
        fs0: Original sampling rate
        fs
        ts (list of arrays): List of indices. These are dimensionless units with an implied
            scaling factor (T_0 = 1/fs) being the sampling interval.
        ds_float64 (list of arrays)
        eggs_float64 (list of arrays)
    """
    marks = parse_marks(marks)
    validate_marks(marks)
    
    fs0, data0 = wavfile.read(path)
    fs, data = resample_data(fs0, data0, int(resample)) if resample else (fs0, data0)
    ds, eggs = extract_pitch_periods_from_data(fs, data, marks)
    ds_float64, eggs_float64 = rescale_to_normalize_ds(ds, eggs)
    ds_float64, eggs_float64 = apply_polarity(ds_float64, eggs_float64, polarity)
    
    ts = [np.arange(len(d)) for d in ds] # Dimensionless units; scaling factor is the
    return fs0, fs, ts, ds_float64, eggs_float64