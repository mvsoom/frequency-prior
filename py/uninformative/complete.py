"""See ./md/complete.md for the logic behind completing the posterior samples"""
import numpy as np
from scipy.stats import invgamma, multivariate_normal
import model
import analyze

import joblib
memory = joblib.Memory('cache', verbose=0)

def sample_sigma(alpha, beta):
    ig = invgamma(alpha, scale=beta)
    z = ig.rvs()
    lpz = ig.logpdf(z)
    
    sigma = np.sqrt(z)
    lp = lpz + np.log(2*sigma) # Add log Jacobian of transformation
    
    return sigma, lp

def sample_bs(b_hats, gs, sigma):
    bs, lps = [], []
    for b_hat, g in zip(b_hats, gs):
        # TODO: We don't need to calculate the inverse of g -- see complete.md
        cov = sigma**2*np.linalg.inv(g)
        mvn = multivariate_normal(mean=b_hat, cov=cov, allow_singular=True)
        b = mvn.rvs()
        bs += [b]
        lps += [mvn.logpdf(b)]
    
    return bs, lps

@memory.cache
def complete_samples(order, data, results, n_jobs=-1):
    samples, logwt = results.samples, results.logwt
    nu, _ = model.order_factors(order, data, np.nan)
    
    def complete_sample(x, lw, beef=1e-6):
        b_hats, gs, chi2s = [], [], []

        for (t, d) in zip(data[1], data[2]):
            G = model.eval_G(t, x, order)

            b_hat, chi2, rank, s = np.linalg.lstsq(G, d, rcond=None)
            g = G.T @ G
            g[np.diag_indices_from(g)] += beef

            b_hats += [b_hat]
            gs += [g]
            chi2s += [float(chi2)]
        
        sigma, lp = sample_sigma(nu/2, np.sum(chi2s)/2)
        bs, lqs = sample_bs(b_hats, gs, sigma)
        
        complete_sample = np.hstack([*bs, x, [sigma]])
        complete_logwt = lw + lp + np.sum(lqs)
        
        return complete_sample, complete_logwt

    with joblib.Parallel(n_jobs=n_jobs) as parallel:
        zipped = parallel(
            joblib.delayed(complete_sample)(x, lw) for x, lw in zip(samples, logwt)
        )

    complete_samples, complete_logwts = zip(*zipped)
    return np.array(complete_samples), np.array(complete_logwts)

def sample_components_and_spectrum(
    complete_samples,
    complete_logwts,
    num_resample,
    order,
    data,
    num_freq
):
    P, Q = order
    n = len(data[1])
    m = P + 2*Q
    
    def alloc_component_samples():
        shapes = [(num_resample, len(data[1][j])) for j in range(n)]
        return [np.empty(shape) for shape in shapes]
    
    def alloc_spectrum_samples():
        shapes = [(num_resample, num_freq) for j in range(n)]
        return [np.empty(shape) for shape in shapes]

    trends = alloc_component_samples()
    periodics = alloc_component_samples()
    spectra = alloc_spectrum_samples()
    
    frequencies = freqspace(num_freq, data[0])
    
    # Downsample to equally-weighted samples
    complete_samples_equal = analyze.resample_equal(
        complete_samples, complete_logwts, num_resample
    )

    for i, complete_sample in enumerate(complete_samples_equal):
        bs, x, sigma = np.split(complete_sample, [n*m, -1])

        bs_pitch_periods = np.split(bs, n)
        for j, (b, t, d) in enumerate(zip(bs_pitch_periods, data[1], data[2])):
            G = model.eval_G(t, x, order)

            trend = G[:,:P] @ b[:P]
            periodic = G[:,P:] @ b[P:]
            spectrum = power_spectrum_dB(b[P:], x, frequencies)

            trends[j][i,:] = trend
            periodics[j][i,:] = periodic
            spectra[j][i,:] = spectrum
    
    # Return equally-weighted samples
    return trends, periodics, spectra

def freqspace(n, fs):
    return np.linspace(0, fs/2, n)

def power_spectrum_dB(b_periodic, x, freqs):
    """Calculate the power spectrum of the impulse response in dB
    
    The impulse response is a sum of decaying sinusoids which are
    parametrized by the periodic amplitudes `b_periodic` and bandwidths
    and frequencies contained in `x`. This function computes the analytical
    magnitude spectrum of that sum and evaluates it at `freqs`. The
    Fourier transform used is the same as (Eq. 1) in the [Wiki page][1].
    For an illustration on a simpler example, see ./FFT_scaling.ipynb.
    
    We calculate this in dB because Gaussian statistics are more meaningful
    in this domain (for example, the power will always be positive.)

        [1]: https://en.wikipedia.org/wiki/Fourier_transform
    """
    b_periodic = b_periodic[:,None]
    x = x[:,None]
    s = (2*np.pi*1j)*freqs[None,:]
    
    b_cos, b_sin = np.split(b_periodic, 2)
    bandwidth, frequency = np.split(x, 2) # Hz
    
    # Rescale
    alpha = np.pi*bandwidth
    omega = 2*np.pi*frequency
    
    # Calculate analytical Fourier transform
    numerator = (alpha + s)*b_cos + omega*b_sin
    denominator = (alpha + s)**2 + omega**2
    transform = np.sum(numerator/denominator, axis=0) # Sum over (bandwidth, frequency) pairs
    
    power_dB = 20*np.log10(np.abs(transform))
    return power_dB