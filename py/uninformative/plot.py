import numpy as np
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
import matplotlib.pyplot as plt
import gvar

def _get_labels(results):
    Q = int(results.samples.shape[1] / 2)
    return [f"$B_{i+1}$" for i in range(Q)] + [f"$F_{i+1}$" for i in range(Q)]

def show_dyplots(results, ylim_quantiles=(0,.99), trace_only=True):
    if trace_only is None: return

    if not trace_only: dyplot.runplot(results)

    # This uses a locally modified version of dyplot.traceplot(). The ylim_quantiles
    # are used to reject samples with a likelihood of zero, i.e. samples with
    # formant frequencies larger than fs/2. There are other ways of dealing with this,
    # such as resampling or sifting out samples with logl = -1e300.
    p, _ = dyplot.traceplot(
        results, show_titles=True, ylim_quantiles=ylim_quantiles, labels = _get_labels(results)
    )
    p.tight_layout() # Somehow this still outputs to Jupyter lab

    if not trace_only: dyplot.cornerplot(results)

def show_modelplots(
    data,
    trends,
    periodics,
    fs,
    num_posterior_samples=25,
    offset=2,
    figsize=(12,2)
):
    def ugly_hack(data, d):
        t = data[1][0]
        dt = t[1] - t[0]
        return np.arange(len(d))*dt*1000 # (msec)
    
    d = np.concatenate(data[2])
    t = ugly_hack(data, d)
    trend = np.concatenate(trends)
    periodic = np.concatenate(periodics)
    f = np.concatenate(fs)
    
    def samples(g):
        s = [gvar.sample(g) for i in range(num_posterior_samples)]
        return np.array(s).T

    def plot_data(i):
        plt.plot(t, d - i*offset, '--', color='black')

    def plot_samples(g, i, color='black', alpha=1/num_posterior_samples):
        plt.plot(t, samples(g) - i*offset, color=color, alpha=alpha)

    width, height = figsize
    plt.figure(figsize=(width, height*3))
    plt.title('Data vs. posterior samples of the model function and its components')
    plt.xlabel('time [msec]')
    plt.ylabel('amplitude [a.u.]')

    # Plot data vs. full model function
    plot_data(0)
    plot_samples(f, 0)
    
    # Plot components
    plot_data(1)
    plot_samples(trend, 1)
    
    plot_data(2)
    plot_samples(periodic, 2)
    
    plt.show()
    
    # Plot normalized glottal flow
    plt.figure(figsize=(width, height))
    plt.title('Data vs. posterior samples of the "glottal flow"')
    plt.xlabel('time [msec]')
    plt.ylabel('amplitude [a.u.]')
    
    gf = np.cumsum(trend)
    gf_mean = gvar.mean(gf)
    scale = gf_mean.max() - gf_mean.min()
    
    plot_data(0)
    plot_samples(gf/scale, 0)

    plt.show()

def show_spectrumplot(
    data,
    spectrum,
    freqs,
    n_pad=None,
    num_posterior_samples=25,
    figsize=(12,4)
):
    plt.figure(figsize=figsize)
    plt.title('Spectrum of data vs. posterior samples of impulse response spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Spectral power (dB)')

    # Calculate power spectrum of data with correct (dt scaling).
    # See ./FFT_scaling.ipynb for details.
    d = np.concatenate(data[2])
    dt = 1/data[0]
    D = np.fft.rfft(d, n_pad)*dt
    D_freq = np.fft.rfftfreq(n_pad if n_pad else len(d), dt)

    plt.plot(D_freq, 20*np.log10(np.abs(D)), '--', color='black')
    
    # Plot posterior samples of pitch-period-averaged power spectrum (already in dB)
    def samples(g):
        s = [gvar.sample(g) for i in range(num_posterior_samples)]
        return np.array(s).T

    plt.plot(freqs, samples(spectrum), color='black', alpha=1/num_posterior_samples)
    
    plt.show()