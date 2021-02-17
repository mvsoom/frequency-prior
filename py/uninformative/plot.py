import numpy as np
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
import matplotlib.pyplot as plt
import gvar

def _get_labels(Q):
    return [f"$B_{i+1}$" for i in range(Q)] + [f"$F_{i+1}$" for i in range(Q)]

def show_dyplots(results, ylim_quantiles=(0,.99), trace_only=True):
    if trace_only is None: return

    if not trace_only: dyplot.runplot(results)

    # This uses a locally modified version of dyplot.traceplot(). The ylim_quantiles
    # are used to reject samples with a likelihood of zero, i.e. samples with
    # formant frequencies larger than fs/2. There are other ways of dealing with this,
    # such as resampling or sifting out samples with logl = -1e300, or using the `thin`
    # or `span` parameters of dyplot.traceplot().
    Q = int(results.samples.shape[1] / 2)
    p, _ = dyplot.traceplot(
        results, show_titles=True, ylim_quantiles=ylim_quantiles, labels = _get_labels(Q)
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

def show_marginalplots(
    samples, weights,
    # Parameters below this line function identically to dyplot.traceplot()
    span=None, quantiles=[0.025, 0.5, 0.975],
    smooth=0.02, dims=None,
    post_color='blue', post_kwargs=None,
    max_n_ticks=5, use_math_text=False,
    labels=None, label_kwargs=None,
    show_titles=False, title_fmt=".2f", title_kwargs=None,
    truths=None, truth_color='red', truth_kwargs=None,
    verbose=False
):
    """Adapted version of dyplot.traceplot() for model-averaged results"""
    
    # Repeat imports necessary for dyplot.traceplot()
    import types
    import matplotlib.pyplot as pl
    from matplotlib.ticker import MaxNLocator, NullLocator
    from matplotlib.colors import LinearSegmentedColormap, colorConverter
    from matplotlib.ticker import ScalarFormatter
    from scipy import spatial
    from scipy.ndimage import gaussian_filter as norm_kde
    from scipy.stats import gaussian_kde
    import warnings
    from dynesty.utils import resample_equal, unitcheck
    from dynesty.utils import quantile as _quantile

    # Initialize values.
    if title_kwargs is None:
        title_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()
    if post_kwargs is None:
        post_kwargs = dict()
    if truth_kwargs is None:
        truth_kwargs = dict()

    # Set defaults.
    post_kwargs['alpha'] = post_kwargs.get('alpha', 0.6)
    truth_kwargs['linestyle'] = truth_kwargs.get('linestyle', 'solid')
    truth_kwargs['linewidth'] = truth_kwargs.get('linewidth', 2)

    # Deal with 1D results. A number of extra catches are also here
    # in case users are trying to plot other results besides the `Results`
    # instance generated by `dynesty`.
    samples = np.atleast_1d(samples)
    if len(samples.shape) == 1:
        samples = np.atleast_2d(samples)
    else:
        assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
        samples = samples.T
    assert samples.shape[0] <= samples.shape[1], "There are more " \
                                                 "dimensions than samples!"

    # Slice samples based on provided `dims`.
    if dims is not None:
        samples = samples[dims]
    ndim, nsamps = samples.shape
    Q = int(ndim/2)

    # Check weights.
    if weights.ndim != 1:
        raise ValueError("Weights must be 1-D.")
    if nsamps != weights.shape[0]:
        raise ValueError("The number of weights and samples disagree!")

    # Determine plotting bounds for marginalized 1-D posteriors.
    if span is None:
        span = [0.999999426697 for i in range(ndim)]
    span = list(span)
    if len(span) != ndim:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = _quantile(samples[i], q, weights=weights)

    # Setting up labels.
    if labels is None:
        labels = _get_labels(Q)

    # Setting up smoothing.
    if (isinstance(smooth, int) or isinstance(smooth, float)):
        smooth = [smooth for i in range(ndim)]

    # Setting up default plot layout.
    fig, axes = pl.subplots(Q, 2, figsize=(12, 3*Q))

    # Plotting.
    for i, x in enumerate(samples):

        # Plot marginalized 1-D posterior.

        # Establish axes.
        ax = axes.T.flatten()[i]
        # Set color(s).
        if isinstance(post_color, str):
            color = post_color
        else:
            color = post_color[i]
        # Setup axes
        ax.set_xlim(span[i])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks))
            ax.yaxis.set_major_locator(NullLocator())
        # Label axes.
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.xaxis.set_major_formatter(sf)
        ax.set_xlabel(labels[i], **label_kwargs)
        # Generate distribution.
        s = smooth[i]
        if isinstance(s, int):
            # If `s` is an integer, plot a weighted histogram with
            # `s` bins within the provided bounds.
            n, b, _ = ax.hist(x, bins=s, weights=weights, color=color,
                              range=np.sort(span[i]), **post_kwargs)
            x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
            y0 = np.array(list(zip(n, n))).flatten()
        else:
            # If `s` is a float, oversample the data relative to the
            # smoothing filter by a factor of 10, then use a Gaussian
            # filter to smooth the results.
            bins = int(round(10. / s))
            n, b = np.histogram(x, bins=bins, weights=weights,
                                range=np.sort(span[i]))
            n = norm_kde(n, 10.)
            x0 = 0.5 * (b[1:] + b[:-1])
            y0 = n
            ax.fill_between(x0, y0, color=color, **post_kwargs)
        ax.set_ylim([0., max(y0) * 1.05])
        # Plot quantiles.
        if quantiles is not None and len(quantiles) > 0:
            qs = _quantile(x, quantiles, weights=weights)
            for q in qs:
                ax.axvline(q, lw=2, ls="dashed", color=color)
            if verbose:
                print("Quantiles:")
                print(labels[i], [blob for blob in zip(quantiles, qs)])
        # Add truth value(s).
        if truths is not None and truths[i] is not None:
            try:
                [ax.axvline(t, color=truth_color, **truth_kwargs)
                 for t in truths[i]]
            except:
                ax.axvline(truths[i], color=truth_color, **truth_kwargs)
        # Set titles.
        if show_titles:
            title = None
            if title_fmt is not None:
                ql, qm, qh = _quantile(x, [0.025, 0.5, 0.975], weights=weights)
                q_minus, q_plus = qm - ql, qh - qm
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
                title = "{0} = {1}".format(labels[i], title)
                ax.set_title(title, **title_kwargs)

    fig.tight_layout()
    return fig, axes

def show_residuals(a, n=5):
    d = np.concatenate(a['data'][2])
    f = np.concatenate(a['fs'])
    sigma = gvar.mean(a['complete_estimates'][-1])
    e = (d - f)/sigma

    plt.figure(figsize=(10,n))
    plt.suptitle("Which one is true Gaussian noise?")
    i = randint(2)
    plt.subplot(1,2,1+i)
    for i in range(n): plt.plot(i+gvar.sample(e)/n)

    plt.subplot(1,2, 1+int(not i))
    for i in range(n): plt.plot(i+randn(len(d))/n)
    plt.show()