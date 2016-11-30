import numpy as np

from scipy.signal import lfilter, hamming
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct

from scikits.talkbox import segment_axis

from scikits.talkbox.features.mel import hz2mel
from scikits.talkbox.features.mfcc import trfbank
import wave
import os

def create_feature(dir0, dir1, nframe=1, threshold=0.6, twin=25, tover=10, nceps=0, tmax=120, verbose=False):
    features = []
    labels = []
    # waves = []
    files0 = [f for f in os.listdir(dir0) if f.endswith(".wav")]
    files1 = [f for f in os.listdir(dir1) if f.endswith(".wav")]
    for filename in files0:
        feature = mel_spectrogram(os.path.join(dir0,filename), nframe, threshold, twin, tover, nceps, tmax, verbose)
        label = np.zeros(feature.shape[0])
        features.append(feature)
        labels.append(label)
        # waves.append(wav)
    for filename in files1:
        feature = mel_spectrogram(os.path.join(dir1,filename), nframe, threshold, twin, tover, nceps, tmax, verbose)
        label = np.ones(feature.shape[0])
        features.append(feature)
        labels.append(label)
        # waves.append(wav)
    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)[:,None]
    # ts = np.concatenate(waves, axis=0)
    return X, y

def create_feature_list(dir0, dir1, nframe=1, threshold=0.6, twin=25, tover=10, nceps=0, tmax=120, maxlen=100, verbose=False):
    features = []
    labels = []
    files0 = [f for f in os.listdir(dir0) if f.endswith(".wav")]
    files1 = [f for f in os.listdir(dir1) if f.endswith(".wav")]
    for filename in files0:
        feature = mel_spectrogram(os.path.join(dir0,filename), nframe, threshold, twin, tover, nceps, tmax, verbose)
        while feature.shape[0] > maxlen:
            features.append(feature[:maxlen])
            feature = feature[maxlen/4:]
            labels.append(0)
        features.append(feature)
        labels.append(0)
    for filename in files1:
        feature = mel_spectrogram(os.path.join(dir1,filename), nframe, threshold, twin, tover, nceps, tmax, verbose)
        while feature.shape[0] > maxlen:
            features.append(feature[:maxlen])
            feature = feature[maxlen/4:]
            labels.append(1)
        features.append(feature)
        labels.append(1)

    return features, labels

def mel_spectrogram(filename, nframe=1, threshold=0.6, twin=25, tover=10, nceps=0, tmax=120, verbose=False):
    wav, fs = wavread(filename)
    nfft = int(twin/1000.0*fs)
    over = int(tover/1000.0*fs)
    if len(wav)/fs > tmax:
        wav = wav[:int(tmax*fs)]
    wav = (wav-wav.mean())/wav.std()
    if nceps:
        _, mel, ceps = mfcc(wav, nfft, nfft, fs, nceps, over)
        feature = ceps
        spec = mel
    else:
        _, mel = mfcc(wav, nfft, nfft, fs, 0, over)
        feature = mel
        spec = mel


    for i in range(nframe - 1):
        feature = np.concatenate((feature[:-1, :], ceps[i + 1:, :]), axis=1)
        spec = np.concatenate((spec[:-1, :], mel[i + 1:, :]), axis=1)

    nbefore = feature.shape[0]
    energy = np.mean(10**spec, axis=1)
    feature = feature[energy > threshold]

    if verbose:
        # print "%s(fs %d)" %(filename, fs)
        print "%d: ts => msp %.3f" %(feature.shape[0], feature.shape[0]/float(nbefore))
    return feature

def wavread(filename):
    wf = wave.open(filename, 'r')
    fs = wf.getframerate()
    x = wf.readframes(wf.getnframes())
    x = np.frombuffer(x, dtype='int16') / 32768.0
    wf.close()
    return x, float(fs)

def mfcc(input, nwin=512, nfft=512, fs=16000, nceps=13, over=256):
    """Compute Mel Frequency Cepstral Coefficients.

    Parameters
    ----------
    input: ndarray
        input from which the coefficients are computed

    Returns
    -------
    ceps: ndarray
        Mel-cepstrum coefficients
    mspec: ndarray
        Log-spectrum in the mel-domain.

    Notes
    -----
    MFCC are computed as follows:
        * Pre-processing in time-domain (pre-emphasizing)
        * Compute the spectrum amplitude by windowing with a Hamming window
        * Filter the signal in the spectral domain with a triangular
        filter-bank, whose filters are approximatively linearly spaced on the
        mel scale, and have equal bandwith in the mel scale
        * Compute the DCT of the log-spectrum

    References
    ----------
    .. [1] S.B. Davis and P. Mermelstein, "Comparison of parametric
           representations for monosyllabic word recognition in continuously
           spoken sentences", IEEE Trans. Acoustics. Speech, Signal Proc.
           ASSP-28 (4): 357-366, August 1980."""

    # MFCC parameters: taken from auditory toolbox
    # over = nwin - step
    # Pre-emphasis factor (to take into account the -6dB/octave rolloff of the
    # radiation at the lips level)
    prefac = 0.97

    #lowfreq = 400 / 3.
    lowfreq = 133.33
    #highfreq = 6855.4976
    linsc = 200/3.
    logsc = 1.0711703

    nlinfil = 13
    nlogfil = 27
    nfil = nlinfil + nlogfil

    w = hamming(nwin, sym=0)

    fbank = trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)[0]

    #------------------
    # Compute the MFCC
    #------------------
    extract = preemp(input, prefac)
    framed = segment_axis(extract, nwin, over) * w

    # Compute the spectrum magnitude
    spec = np.abs(fft(framed, nfft, axis=-1))
    # Filter the spectrum through the triangle filterbank
    mspec = np.log10(np.dot(spec, fbank.T))
    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    if nceps:
        ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, :nceps]
        return spec, mspec, ceps
    else:
        return spec, mspec

def preemp(input, p):
    """Pre-emphasis filter."""
    return lfilter([1., -p], 1, input)

# def trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
#     """Compute triangular filterbank for MFCC computation."""
#     # Total number of filters
#     nfilt = nlinfilt + nlogfilt
#
#     #------------------------
#     # Compute the filter bank
#     #------------------------
#     # Compute start/middle/end points of the triangular filters in spectral
#     # domain
#     freqs = np.zeros(nfilt+2)
#     freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
#     freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
#     heights = 2./(freqs[2:] - freqs[0:-2])
#
#     # Compute filterbank coeff (in fft domain, in bins)
#     fbank = np.zeros((nfilt, nfft))
#     # FFT bins (in Hz)
#     nfreqs = np.arange(nfft) / (1. * nfft) * fs
#     for i in range(nfilt):
#         low = freqs[i]
#         cen = freqs[i+1]
#         hi = freqs[i+2]
#
#         lid = np.arange(np.floor(low * nfft / fs) + 1,
#                         np.floor(cen * nfft / fs) + 1, dtype=np.int)
#         lslope = heights[i] / (cen - low)
#         rid = np.arange(np.floor(cen * nfft / fs) + 1,
#                         np.floor(hi * nfft / fs) + 1, dtype=np.int)
#         rslope = heights[i] / (hi - cen)
#         fbank[i][lid] = lslope * (nfreqs[lid] - low)
#         fbank[i][rid] = rslope * (hi - nfreqs[rid])
#
#     return fbank, freqs

# if __name__ == '__main__':
#     extract = loadmat('extract.mat')['extract']
#     ceps = mfcc(extract)
