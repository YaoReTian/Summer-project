import numpy as np
from scipy.signal import butter, filtfilt

#signal = np.random.rand(1000)  # Example signal

# making a trial waveform for testing
"""
f = 440
omega = 2 * np.pi * f
waveform = lambda t: np.sin(omega * t)
"""

config = {
    'awgn':          {'snr': 20},

    'impulse':       {'prob': 0.001, 'magnitude': 5},

    'multipath':     {'delays': [50, 120], 'attenuations': [0.5, 0.25]},

    'lowpass':       {'low_cutoff': 4000, 'high_cutoff': None},

    'quantise':      {'bits': 8},

    'interference':  {'freq': 1000, 'strength': 0.3}

}

def add_awgn(snr):
    def channel(waveform):
        def noisy_waveform(t):
            x = waveform(t)
            power_signal = np.mean(x**2) # calculating the power of the signal from definition of power for a signal
            snr_linear = 10 ** (snr / 10.0)
            noise_power = power_signal / snr_linear # calculating the power of the noise using the SNR formula rearranged to find noise power
            noise = np.random.normal(0, np.sqrt(noise_power), np.shape(t)) # generating the gaussian noise with the calculated power
            return x + noise # adding the noise to the original signal to create the noisy signal
        return noisy_waveform
    return channel

def add_impulse_noise(prob=0.001, magnitude=5):
    def channel(waveform):
        def noisy_waveform(t):
            x = waveform(t)
            t_arr = np.atleast_1d(t) # ensure t is an array for consistent processing, even if a single time point is given
            mask = np.random.rand(*t_arr.shape) < prob # creating a boolean mask where True indicates where the impulse noise will be added, based on the specified probability
            signs = np.random.choice([-1, 1], size=t_arr.shape) # randomly choosing whether the impulse noise will be positive or negative
            return x + mask * signs * magnitude
        return noisy_waveform
    return channel

def multipath(delays=[0.001, 0.002], attenuations=[0.5, 0.25]):
    # note delays now in seconds, since we are dealing with waveforms, we will convert these to sample delays based on the sampling frequency when we apply the channel
    def channel(waveform):
        def multipath_waveform(t):
            signal = waveform(t)
            result = signal.copy() # start with the original signal as the base of the result
            for delay, att in zip(delays, attenuations):
                result = result + att * waveform(t - delay) # add the delayed and attenuated version of the signal to the result to simulate the multipath effect
            return result
        return multipath_waveform
    return channel

"""
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
"""
#analogue wouldn't make sense either, since we are working with discrete signals, so we will keep the cutoff frequencies in Hz and convert to normalized frequencies within the function

"""
def quantise(signal, bits=8):

    levels = 2**bits # calculate the number of quantisation levels based on the number of bits (e.g., 8 bits gives 256 levels)

    sig_min, sig_max = signal.min(), signal.max()

    normalised = (signal - sig_min) / (sig_max - sig_min) # normalise the signal to the range [0, 1] by subtracting the minimum value and dividing by the range (max - min)

    quantised = np.round(normalised * (levels - 1)) / (levels - 1) # quantise the normalised signal to the specified number of levels by scaling to the range [0, levels-1], rounding to the nearest integer, and then scaling back to [0, 1]

    return quantised * (sig_max - sig_min) + sig_min  # scale the quantised signal back to the original range by multiplying by the range (max - min) and adding the minimum value
"""

def add_sinusoidal_interference(freq=1000, strength=0.3):
    def channel(waveform):
        def interfered_waveform(t):
            return waveform(t) + strength * np.sin(2 * np.pi * freq * t) # add a sinusoidal interference signal with the specified frequency and strength to the original waveform
        return interfered_waveform
    return channel


def build_channel(*transforms):
    def channel(waveform):
        result = waveform
        for transform in transforms:
            result = transform(result)
        return result
    return channel


"""
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
"""
