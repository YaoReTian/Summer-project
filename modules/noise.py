import numpy as np
from scipy.signal import butter, filtfilt

signal = np.random.rand(1000)  # Example signal


config = {
    'awgn':          {'snr_db': 20}, 

    'impulse':       {'prob': 0.001, 'magnitude': 5}, 

    'multipath':     {'delays': [50, 120], 'attenuations': [0.5, 0.25]}, 

    'lowpass':       {'low_cutoff': 4000, 'high_cutoff': None}, 

    'quantise':      {'bits': 8}, 

    'interference':  {'freq': 1000, 'strength': 0.3} 

} 

def add_awgn(signal, snr_db): 

    snr = 10**(snr_db/10) # using the SNR formula to change SNR from dB to linear scale

    power_signal = np.mean(signal**2) # calculating the power of the signal from definition of power for a signal

    noise_power = power_signal / snr # calculating the power of the noise using the SNR formula rearranged to find noise power

    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape) # generating the gaussian noise with the calculated power

    return signal + noise  

def add_impulse_noise(signal, prob=0.001, magnitude=5): 

    mask = np.random.rand(*signal.shape) < prob # creating a boolean mask where True indicates where the impulse noise will be added, based on the specified probability

    signs = np.random.choice([-1, 1], size=signal.shape) # randomly choosing whether the impulse noise will be positive or negative

    return signal + mask * signs * magnitude  

def multipath(signal, delays=[50, 120], attenuations=[0.5, 0.25]): 

    result = signal.copy() # start with the original signal as the base for the multipath effect

    for delay, att in zip(delays, attenuations): 

        echo = np.zeros_like(signal) # create an array of zeros to hold the delayed and attenuated signal

        echo[delay:] = att * signal[:-delay] # create the echo by taking the original signal, shifting it by the delay, and scaling it by the attenuation factor

        result += echo 

    return result 

# Improved: validate cutoffs and support low/high/band filters
def apply_bandpass_filter(signal, low_cutoff, high_cutoff, fs):

    nyq = fs / 2 # calculate the Nyquist frequency (half the sampling rate)

    if low_cutoff == None and high_cutoff == None:
        raise ValueError("At least one of low_cutoff or high_cutoff must be provided")

    # Lowpass only
    if high_cutoff == None:
        Wn = low_cutoff / nyq
        if not 0 < Wn < 1:
            raise ValueError(f"low_cutoff must be between 0 and Nyquist ({nyq}). Got {low_cutoff}.")
        b, a = butter(4, Wn, btype='low') # design a 4th order Butterworth lowpass filter with the normalized cutoff frequency Wn

    # Highpass only
    elif low_cutoff == None:
        Wn = high_cutoff / nyq
        if not 0 < Wn < 1:
            raise ValueError(f"high_cutoff must be between 0 and Nyquist ({nyq}). Got {high_cutoff}.")
        b, a = butter(4, Wn, btype='high') # design a 4th order Butterworth highpass filter with the normalized cutoff frequency Wn

    # Bandpass
    else:
        low_norm = low_cutoff / nyq
        high_norm = high_cutoff / nyq
        if not (0 < low_norm < high_norm < 1):
            raise ValueError(f"For bandpass require 0 < low_cutoff < high_cutoff < Nyquist ({nyq}). Got {low_cutoff}, {high_cutoff}.")
        b, a = butter(4, [low_norm, high_norm], btype='bandpass')  # design a 4th order Butterworth bandpass filter with the normalized cutoff frequencies low_norm and high_norm

    return filtfilt(b, a, signal)

def quantise(signal, bits=8): 

    levels = 2**bits # calculate the number of quantisation levels based on the number of bits (e.g., 8 bits gives 256 levels)

    sig_min, sig_max = signal.min(), signal.max() 

    normalised = (signal - sig_min) / (sig_max - sig_min) # normalise the signal to the range [0, 1] by subtracting the minimum value and dividing by the range (max - min)

    quantised = np.round(normalised * (levels - 1)) / (levels - 1) # quantise the normalised signal to the specified number of levels by scaling to the range [0, levels-1], rounding to the nearest integer, and then scaling back to [0, 1]

    return quantised * (sig_max - sig_min) + sig_min  # scale the quantised signal back to the original range by multiplying by the range (max - min) and adding the minimum value

def add_sinusoidal_interference(signal, fs, freq=1000, strength=0.3): 

    t = np.arange(len(signal)) / fs # create a time vector based on the length of the signal and the sampling frequency (fs) to represent the time at each sample point

    interferer = np.sin(2 * np.pi * freq * t) # generate a sinusoidal interference signal at the specified frequency (freq) by calculating the sine of 2*pi*freq*t, which creates a sinusoid that oscillates at the desired frequency over time

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