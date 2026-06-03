from PIL import Image
import numpy as np
import noise
from pyldpc import make_ldpc, encode, decode, get_message 
import time
import json 
import loss
"""INSTEAD OF IMPORTING LDPC_ALGORITHM, GOING TO COPY ALL OF IT HERE """

#start clock and define function to check time
start=time.time()
def check_time(start):
    end=time.time()
    return end-start
    

#Predetermined generator matrix
with open('generator.txt', 'r') as file:
    G =np.array(json.load(file))
    
with open('hamming_matrix.txt', 'r') as file:
    H =np.array(json.load(file))

#add padding to the remainder array of w % k
def pad_bits_to_block_size(bits, k=904):

    bits = np.asarray(bits, dtype=int).ravel()
#check if array is binary
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Input bits must only contain 0 or 1.")

    original_len = len(bits)
    pad_len = (-original_len) % k

    if pad_len:
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=int)])

    return bits, original_len, pad_len

def split_into_blocks(bits, k=904):
    """
    Split a bit array into blocks of length k, with zero padding if needed.
    """
    padded_bits, original_len, pad_len = pad_bits_to_block_size(bits, k)
    """
    -1 here just implies numpy can figure out the shape given the other parameter so it is 
    compatible with previous array
    """
    blocks = padded_bits.reshape(-1, k)
    return blocks, original_len, pad_len

def encode_block(block, G):
    """
    Encodes one message block using generator matrix G over     
    """
    block = np.asarray(block, dtype=int).ravel()
    G = np.asarray(G, dtype=int)

    
    codeword = np.asarray((G @ block) % 2).astype(int).ravel()
    return np.asarray(codeword, dtype=int).ravel()

def encode_bit_array_ldpc(bits, G, k=904, return_blocks=False):
    """
    Encode a bit array of any length using LDPC generator matrix G
    """
    n, _ = G.shape
    blocks, original_len, pad_len = split_into_blocks(bits, k=k)

    encoded_blocks = np.empty((blocks.shape[0], n), dtype=np.uint8)

    for i in range(blocks.shape[0]):
        encoded_blocks[i] = encode_block(blocks[i], G)

    result = {
        "encoded_bits": encoded_blocks.ravel(),
        "original_length": original_len,
        "pad_length": pad_len,
        "num_blocks": blocks.shape[0],
        "message_block_size": k,
        "codeword_length": n,
    }

    if return_blocks:
        result["encoded_blocks"] = encoded_blocks

    return result
"""TIME FOR DECODING ALGO"""
def split_codewords(encoded_bits, n):
    """
    Split a 1D encoded bit array into blocks of codeword length n.
    """
    encoded_bits = np.asarray(encoded_bits).ravel()

    if len(encoded_bits) % n != 0:
        raise ValueError(
            f"Encoded bitstream length {len(encoded_bits)} is not divisible by codeword length n={n}."
        )

    return encoded_bits.reshape(-1, n)
def decode_ldpc_blocks(channel_values, H, G, original_length,codeword_length, snr, maxiter):
    """
    Decode a stream of LDPC channel observations block-by-block.
    """

    codeword_blocks = split_codewords(channel_values, codeword_length)

    decoded_blocks = []

    for i, y in enumerate(codeword_blocks):
        d = decode(H, y, snr,maxiter=maxiter)
        msg = get_message(G,d)
        print(f"{d[:10]}...")
        print(f"It took {check_time(start)} for iteration {i} to finish")
        msg = np.asarray(msg, dtype=int).ravel()
        decoded_blocks.append(msg)

    decoded_bits = np.concatenate(decoded_blocks)
    decoded_bits = decoded_bits[:original_length]

    return decoded_bits





"""
define some useful functions for ldpc algorithm

"""
#converts a 1D array of integers to a 1D array of bits
def ints_to_flat_bits(arr, width=8):
    return [(int(n) >> i) & 1 for n in arr for i in range(width - 1, -1, -1)]

#converts a 1D array of bits to a 1D array of integers
def bits_to_ints(bits, width=8):
    if len(bits) % width != 0:
        raise ValueError(f"Number of bits must be a multiple of {width}")

    result = []
    for i in range(0, len(bits), width):
        chunk = bits[i:i + width]
        value = 0
        for bit in chunk:
            value = (value << 1) | int(bit)
        result.append(value)
    return np.array(result)

#define ldpc encoding algorithm 
def ldpc_encode(arr,G,k=904,return_blocks=False):
    #convert to bit array for ldpc
    binary_arr=np.array(ints_to_flat_bits(arr))
    print(f"It took {check_time(start)} time to map integers to bits")
    #iterate generator matrix
    encoded_arr = encode_bit_array_ldpc(binary_arr, G, k=904, return_blocks=True)
    print(f"It took {check_time(start)} time to introduce parity bits to each part")
    flat_binary_arr = encoded_arr["encoded_blocks"].flatten()
    
    #print some params out to check
    print("Original length:", encoded_arr["original_length"])
    print("Pad length:", encoded_arr["pad_length"])
    print("Number of blocks:", encoded_arr["num_blocks"])
    print("Codeword Length:", encoded_arr["codeword_length"])
    print("Encoded blocks shape:", encoded_arr["encoded_blocks"].shape)
    #return to int array with added parity bits (G matrix specifically made so parity matrix is multiple of 8)
    return flat_binary_arr,binary_arr, encoded_arr["original_length"], encoded_arr["codeword_length"]
    





img = Image.open('smalljwphoto.jpeg').convert("RGB") # convert to RGB to ensure we have 3 channels
img_array = np.array(img, dtype=int) 
original_shape = img_array.shape # store original shape for later
original_signal = img_array.flatten() # flatten the 3D image array to a 1D signal for processing
print(f"Original signal shape: {original_signal.shape}")
#encoding ldpc algorithm 

flat_signal,binary_arr, original_length,codeword_length  = ldpc_encode(original_signal,G,k=904,return_blocks=True)
print(f"It took {check_time(start)} time to finish ldpc encoding process")

# get the length of the signal for later use
N = len(flat_signal) 
print(f"Original signal length: {N}")
#normalize using soft trim 
def bits_to_analog_levels(bits, low=0.1, high=0.9):
    bits = np.asarray(bits, dtype=np.uint8)
    return np.where(bits == 0, low, high).astype(np.float64)

tx_signal = bits_to_analog_levels(flat_signal, low=0.02, high=0.98)


"""
NOISE STARTS HERE
"""
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
waveform, Ts, N = make_waveform(tx_signal, sample_rate=sample_rate)

# Time axis: one point per original sample (Nyquist-consistent sampling)
t = np.linspace(0, N * Ts, N, endpoint=False)

channel1 = noise.build_channel(
    noise.add_awgn(snr=5)
)

channel2 = noise.build_channel(
    noise.add_sinusoidal_interference(freq=100, strength=0.2)
)

channel3 = noise.build_channel(
    noise.add_impulse_noise(prob=0.001, magnitude=0.3)
)

channel4 = noise.build_channel(
    noise.multipath(delays=[0.001, 0.002], attenuations=[0.5, 0.25])
)

channel5 = noise.build_channel(
    noise.add_impulse_noise(prob=0.001, magnitude=0.3),
    noise.multipath(delays=[0.001, 0.002], attenuations=[0.5, 0.25]),
    noise.add_sinusoidal_interference(freq=1000, strength=0.5),
    noise.add_awgn(snr=5)
)


# Apply channel to waveform — produces a new callable
noisy_waveform = channel1(waveform) # change number to get different noise types

# Sample the noisy waveform at the original time points
received_signal = noisy_waveform(t)  # shape: (N,)

def analog_to_soft(received, low=0.10, high=0.90):
    midpoint = 0.5 * (low + high)
    scale = 2.0 / (high - low)
    return (received - midpoint) * scale

soft_signal = analog_to_soft(received_signal, low=0.02, high=0.98)
#compute the equivalent so snr can be clearly calculated
clean_soft_signal = analog_to_soft(tx_signal, low=0.02, high=0.98)
print(f"the output soft signal is {soft_signal}")
print(f"It took {check_time(start)} time to finish noise introduction process")


"""NOISE ENDS HERE"""



# Time to decode using LDPC 
#Find the actual snr
snr=loss.snr(soft_signal,clean_soft_signal)
print(f"the real snr is {snr}")

#define ldpc decoding algorithm with noise introduced 
def ldpc_decode(noisy_arr, H,G ,snr, original_length=original_length,codeword_length=codeword_length, maxiter=200):
    decoded_bits = decode_ldpc_blocks(noisy_arr, H, G, original_length,codeword_length, snr, maxiter)
    return decoded_bits

recovered_bits=ldpc_decode(-soft_signal,H,G,snr,original_length,codeword_length,maxiter=500)

print(f"It took {check_time(start)} time to finish ldpc decoding process")
print(f"recovered_bits shape: {recovered_bits.shape}...")

#check ber before converting
ber= loss.ber(np.array(recovered_bits,dtype=int).flatten(),np.array(binary_arr,dtype=int).flatten())
print("ber:",ber)

#change back to integers
recovered_pixels = bits_to_ints(recovered_bits)


# Reshape to original image dimensions
recovered_array = original_signal.reshape(original_shape)

# Save and display
corrupted_img = Image.fromarray(recovered_array, mode="RGB")
corrupted_img.save("corrupted_analogue.jpg")
corrupted_img.show()
