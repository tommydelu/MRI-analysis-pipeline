import pywt
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import detrend

def applyAndCompareWTMethods(signal: np.ndarray, sampling_period: float, 
                             freqs_range: list[float, float], region_name: str,
                             shared_scale:bool=True) -> None:

    wavelet_type = ['mexh', 'morl', 'cmor1.5-1.0', 'gaus1', 'cgau1', 'shan1.5-1.0']

    # It is necessary to transform the frequency range in a scale range
    f_min = freqs_range[0]
    f_max = freqs_range[1]
    time_seconds = np.arange(len(signal)) * sampling_period

    # Problema: la wavelet è implementata con FFT --> convoluzione --> quindi è compreso zero padding
    # Se il segnale inizia e finisce con una risposta molto forte, ai lti avrò molto contrasto --> wavelet interpreta come tanta energia
    # energia talmente alta che tutto il resto è molto + basso quindi nel plot vedo tutto nero
    # soluzione: uso detrend ovvero centro il segnale rispetto allo 0, così non parto e finisco con picchi altissimi

    signal = detrend(signal)
    # Normalizzazione per confrontare segnali diversi, avere coefficienti + contenuti
    signal= (signal - np.mean(signal)) / np.std(signal)
    

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle(f"{region_name} - Wavelet Scalograms (CWT)", fontsize=14, fontweight="bold")    
    axs = axs.flatten()

    cwt_results = []
    global_max = 0.0
    
    for name in wavelet_type:

        scale_min = pywt.frequency2scale(name, f_max * sampling_period)
        scale_max = pywt.frequency2scale(name, f_min * sampling_period)

        scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num=64)
        
        cwtmatr, freqs = pywt.cwt(signal, scales=scales, wavelet=name, sampling_period=sampling_period)
        cwt_magnitude = np.abs(cwtmatr)
        cwt_results.append((name, cwt_magnitude, freqs, scale_min, scale_max))
        global_max = max(global_max, cwt_magnitude.max())

    im = None
    for i, (name, mag, freqs, s_min, s_max) in enumerate(cwt_results):
        vmax = global_max if shared_scale else None

        im = axs[i].pcolormesh( # to plot correctly the logaritmic scale
            time_seconds,
            freqs,
            mag,
            cmap="inferno",
            shading="auto",
            vmax=vmax,
        )

        axs[i].set_title(f"{name} (Scales: [{s_min:.1f}, {s_max:.1f}])")
        if i >= 3:
            axs[i].set_xlabel("Time (s)")
        if i % 3 == 0:
            axs[i].set_ylabel("Frequency (Hz)")

    cbar_label = "WT Magnitude"
    fig.colorbar(im, ax=axs.tolist(), label=cbar_label, shrink=0.8)
    plt.show()