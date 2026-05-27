import numpy as np
from scipy.signal import butter, filtfilt

signal = np.random.rand(1000)  # Example signal


config = { 
    'awgn':          {'snr_db': 20}, 

    'impulse':       {'prob': 0.001, 'magnitude': 5}, 

    'multipath':     {'delays': [50, 120], 'attenuations': [0.5, 0.25]}, 

    'lowpass':       {'low_cutoff': 4000, 'high_cutoff': 0}, 

    'quantise':      {'bits': 8}, 

    'interference':  {'freq': 1000, 'strength': 0.3} 

} 

def add_awgn(signal, snr_db): 

    snr = 10**(snr_db/10) 

    power_signal = np.mean(signal**2) 

    noise_power = power_signal / snr 

    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape) 

    return signal + noise  

def add_impulse_noise(signal, prob=0.001, magnitude=5): 

    mask = np.random.rand(*signal.shape) < prob 

    signs = np.random.choice([-1, 1], size=signal.shape) 

    return signal + mask * signs * magnitude  

def multipath(signal, delays=[50, 120], attenuations=[0.5, 0.25]): 

    result = signal.copy() 

    for delay, att in zip(delays, attenuations): 

        echo = np.zeros_like(signal) 

        echo[delay:] = att * signal[:-delay] 

        result += echo 

    return result  

# TO FIX
def apply_bandpass_filter(signal, low_cutoff, high_cutoff, fs): 

    nyq = fs / 2 

    b, a = butter(4, [low_cutoff / nyq, high_cutoff / nyq], btype='bandpass') 

    return filtfilt(b, a, signal)  

def quantise(signal, bits=8): 

    levels = 2**bits 

    sig_min, sig_max = signal.min(), signal.max() 

    normalised = (signal - sig_min) / (sig_max - sig_min) 

    quantised = np.round(normalised * (levels - 1)) / (levels - 1) 

    return quantised * (sig_max - sig_min) + sig_min  

def add_sinusoidal_interference(signal, fs, freq=1000, strength=0.3): 

    t = np.arange(len(signal)) / fs 

    interferer = np.sin(2 * np.pi * freq * t) 

    return signal + strength * interferer  

 
def apply_channel(signal, fs, config): 

    s = signal.copy().astype(float)  # work on a copy, ensure float type 
 
    if config.get('awgn'): 

        s = add_awgn(s, snr_db=config['awgn']['snr_db']) 

    if config.get('impulse'): 

        s = add_impulse_noise(s, **config['impulse']) 

    if config.get('multipath'): 

        s = multipath(s, **config['multipath']) 

    if config.get('lowpass'): 

        s = apply_bandpass_filter(s, **config['lowpass'], fs=fs) 

    if config.get('quantise'): 

        s = quantise(s, **config['quantise']) 

    if config.get('interference'): 

        s = add_sinusoidal_interference(s, fs=fs, **config['interference']) 

    return s  