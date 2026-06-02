from PIL import Image
import numpy as np
import noise


img = Image.open('data/rick.jpg').convert("RGB") # convert to RGB to ensure we have 3 channels
img_array = np.array(img, dtype=np.float32) # convert to float for processing
original_shape = img_array.shape # store original shape for later

#flatten to 1D and normalise to [-1, 1]
flat_signal = img_array.flatten() # flatten the 3D image array to a 1D signal for processing
flat_signal = (flat_signal / 127.5) - 1.0 # normalise the signal to the range [-1, 1] by dividing by 127.5 and subtracting 1

N = len(flat_signal) # get the length of the signal for later use
print(f"Original signal length: {N}")

def make_waveform(signal, sample_rate=8000): 
    Ts = 1 / sample_rate # calculate the sampling period from the sampling frequency
    N = len(signal) # get the length of the signal

    def waveform(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64)) # ensure t is a numpy array for vectorised operations
        indices = np.floor(t_arr / Ts).astype(int) % N # calculate the corresponding indices in the signal for each time value, using floor to get the correct index and modulo for wrapping around
        indices = np.clip(indices, 0, N - 1) # ensure indices are within bounds of the signal array
        return signal[indices] # return the waveform values corresponding to the input time values
    
    return waveform, Ts, N

sample_rate = 8000  # Hz — increase for finer resolution
waveform, Ts, N = make_waveform(flat_signal, sample_rate=sample_rate)

# Time axis: one point per original sample (Nyquist-consistent sampling)
t = np.linspace(0, N * Ts, N, endpoint=False)

channel1 = noise.build_channel(
    noise.add_awgn(snr_db=5)
)

channel2 = noise.build_channel(
    noise.add_sinusoidal_interference(freq=1000, strength=0.5)
)

channel3 = noise.build_channel(
    noise.add_impulse_noise(prob=0.1, magnitude=0.9)
)

channel4 = noise.build_channel(
    noise.multipath(delays=[0.001, 0.002], attenuations=[0.5, 0.25])
)

# Apply channel to waveform — produces a new callable
noisy_waveform = channel4(waveform) # change number to get different noise types

# Sample the noisy waveform at the original time points
received_signal = noisy_waveform(t)  # shape: (N,)

# Clip to valid normalised range, then denormalise back to [0, 255]
received_clipped = np.clip(received_signal, -1.0, 1.0)
recovered_pixels = ((received_clipped + 1.0) * 127.5).astype(np.uint8)

# Reshape to original image dimensions
recovered_array = recovered_pixels.reshape(original_shape)

# Save and display
corrupted_img = Image.fromarray(recovered_array, mode="RGB")
corrupted_img.save("data/corrupted_analogue4.jpg")
corrupted_img.show()