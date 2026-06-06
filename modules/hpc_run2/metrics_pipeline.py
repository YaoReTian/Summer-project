from PIL import Image
import numpy as np
import noise
from pyldpc import make_ldpc, encode, decode, get_message
import time
import json
import loss
from skimage.metrics import structural_similarity as ssim
import metrics
import matplotlib.pyplot as plt

"""INSTEAD OF IMPORTING LDPC_ALGORITHM, GOING TO COPY ALL OF IT HERE"""

# start clock and define function to check time
start = time.time()

def check_time(start):
    end = time.time()    
    return end - start


# Predetermined generator matrix
with open('generator.txt', 'r') as file:
    G = np.array(json.load(file))

with open('hamming_matrix.txt', 'r') as file:
    H = np.array(json.load(file))


# add padding to the remainder array of w % k
def pad_bits_to_block_size(bits, k=904):
    bits = np.asarray(bits, dtype=int).ravel()

    # check if array is binary
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
    Encodes one message block using generator matrix G
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


def decode_ldpc_blocks(channel_values, H, G, original_length, codeword_length, snr, maxiter):
    """
    Decode a stream of LDPC channel observations block-by-block.
    """
    codeword_blocks = split_codewords(channel_values, codeword_length)
    decoded_blocks = []

    for i, y in enumerate(codeword_blocks):
        d = decode(H, y, snr, maxiter=maxiter)
        msg = get_message(G, d)
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

# converts a 1D array of integers to a 1D array of bits
def ints_to_flat_bits(arr, width=8):
    return [(int(n) >> i) & 1 for n in arr for i in range(width - 1, -1, -1)]


# converts a 1D array of bits to a 1D array of integers
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


# define ldpc encoding algorithm
def ldpc_encode(arr, G, k=904, return_blocks=False):
    # convert to bit array for ldpc
    binary_arr = np.array(ints_to_flat_bits(arr))
    print(f"It took {check_time(start)} time to map integers to bits")

    # iterate generator matrix
    encoded_arr = encode_bit_array_ldpc(binary_arr, G, k=k, return_blocks=True)
    print(f"It took {check_time(start)} time to introduce parity bits to each part")

    flat_binary_arr = encoded_arr["encoded_blocks"].flatten()

    # print some params out to check
    print("Original length:", encoded_arr["original_length"])
    print("Pad length:", encoded_arr["pad_length"])
    print("Number of blocks:", encoded_arr["num_blocks"])
    print("Codeword Length:", encoded_arr["codeword_length"])
    print("Encoded blocks shape:", encoded_arr["encoded_blocks"].shape)

    return flat_binary_arr, binary_arr, encoded_arr["original_length"], encoded_arr["codeword_length"]


def bits_to_analog_levels(bits, low=0.1, high=0.9):
    bits = np.asarray(bits, dtype=np.uint8)
    return np.where(bits == 0, low, high).astype(np.float64)


def make_waveform(signal, sample_rate=8000):
    Ts = 1 / sample_rate
    N = len(signal)

    def waveform(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
        indices = np.floor(t_arr / Ts).astype(int) % N
        indices = np.clip(indices, 0, N - 1)
        return signal[indices]

    return waveform, Ts, N


def analog_to_soft(received, low=0.10, high=0.90):
    midpoint = 0.5 * (low + high)
    scale = 2.0 / (high - low)
    return (received - midpoint) * scale


def ldpc_decode(noisy_arr, H, G, snr, original_length, codeword_length, maxiter=200):
    decoded_bits = decode_ldpc_blocks(
        noisy_arr, H, G, original_length, codeword_length, snr, maxiter
    )
    return decoded_bits


# -------------------------------
# CHANNEL BUILDERS FOR SWEEPING
# -------------------------------

def make_channel(channel_name, param):
    if channel_name == "awgn":
        return noise.build_channel(
            noise.add_awgn(snr=param)
        )

    elif channel_name == "sinusoidal":
        return noise.build_channel(
            noise.add_sinusoidal_interference(freq=param, strength=0.3)
        )

    elif channel_name == "impulse":
        return noise.build_channel(
            noise.add_impulse_noise(prob=param, magnitude=0.3)
        )

    elif channel_name == "multipath":
        return noise.build_channel(
            noise.multipath(delays=[0.001, 0.002], attenuations=[param, param / 2])
        )

    elif channel_name == "combined":
        return noise.build_channel(
            noise.add_impulse_noise(prob=min(0.0005 * param, 0.01), magnitude=0.3),
            noise.multipath(delays=[0.001, 0.002], attenuations=[min(0.1 * param, 0.9), min(0.05 * param, 0.5)]),
            noise.add_sinusoidal_interference(freq=1000, strength=min(0.1 * param, 1.0)),
            noise.add_awgn(snr=max(param, 0.1))
        )

    else:
        raise ValueError(f"Unknown channel_name: {channel_name}")


# -------------------------------
# SINGLE EXPERIMENT
# -------------------------------

def run_channel_experiment(img_array, G, H, channel_name, param, sample_rate=8000, maxiter=500):
    original_shape = img_array.shape
    original_signal = img_array.flatten()

    print(f"\nRunning channel={channel_name}, param={param}")

    # encoding
    flat_signal, binary_arr, original_length, codeword_length = ldpc_encode(
        original_signal, G, k=904, return_blocks=True
    )
    print(f"It took {check_time(start)} time to finish ldpc encoding process")

    N = len(flat_signal)
    print(f"Original signal length: {N}")

    tx_signal = bits_to_analog_levels(flat_signal, low=-1, high=1)

    # waveform prep
    waveform, Ts, N = make_waveform(tx_signal, sample_rate=sample_rate)
    t = np.linspace(0, N * Ts, N, endpoint=False)

    # build selected channel
    selected_channel = make_channel(channel_name, param)

    # Apply channel
    noisy_waveform = selected_channel(waveform)

    # Sample noisy waveform
    received_signal = noisy_waveform(t)

    soft_signal = analog_to_soft(received_signal, low=-1, high=1)
    clean_soft_signal = analog_to_soft(tx_signal, low=-1, high=1)

    print(f"It took {check_time(start)} time to finish noise introduction process")

    # Find actual snr
    snr = loss.snr(clean_soft_signal, soft_signal)
    print(f"the real snr is {snr}")

    # decode
    recovered_bits = ldpc_decode(
        -soft_signal,
        H,
        G,
        snr,
        original_length=original_length,
        codeword_length=codeword_length,
        maxiter=maxiter
    )

    print(f"It took {check_time(start)} time to finish ldpc decoding process")

    recovered_bits = np.array(recovered_bits, dtype=int).flatten()
    binary_arr = np.array(binary_arr, dtype=int).flatten()

    # ber before converting
    ber_before = loss.ber(recovered_bits, binary_arr)
    print("ber:", ber_before)

    # return to integers
    recovered_pixels = bits_to_ints(recovered_bits)

    # Reshape to original image dimensions
    recovered_array = recovered_pixels.reshape(original_shape)

    # METRICS
    ber, ssim_score, snr_metric, psnr = metrics.metric_pipeline(
        img_array,
        recovered_array,
        clean_soft_signal,
        soft_signal
    )

    print(f"ber:{ber}, ssim:{ssim_score}, snr:{snr_metric}, psnr:{psnr}")

    return {
        "param": param,
        "ber": ber,
        "ssim": ssim_score,
        "snr": snr_metric,
        "psnr": psnr,
        "recovered_array": recovered_array
    }


# -------------------------------
# SWEEP FUNCTION
# -------------------------------

def sweep_channel(img_array, G, H, channel_name, param_values, sample_rate=8000, maxiter=500):
    results = {
        "param": [],
        "ber": [],
        "ssim": [],
        "snr": [],
        "psnr": []
    }

    for param in param_values:
        out = run_channel_experiment(
            img_array=img_array,
            G=G,
            H=H,
            channel_name=channel_name,
            param=param,
            sample_rate=sample_rate,
            maxiter=maxiter
        )

        results["param"].append(out["param"])
        results["ber"].append(out["ber"])
        results["ssim"].append(out["ssim"])
        results["snr"].append(out["snr"])
        results["psnr"].append(out["psnr"])

    return results


# -------------------------------
# PLOTTING 16 GRAPHS
# -------------------------------

def plot_16_graphs(all_results):
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle("LDPC Image Transmission Metrics Across 4 Noise Channels", fontsize=18)

    channel_order = ["awgn", "sinusoidal", "impulse", "multipath"]
    metric_order = ["ber", "ssim", "snr", "psnr"]

    for i, channel_name in enumerate(channel_order):
        result = all_results[channel_name]
        x = result["param"]

        for j, metric_name in enumerate(metric_order):
            ax = axes[i, j]
            ax.plot(x, result[metric_name], marker='o')
            ax.set_title(f"{channel_name.upper()} - {metric_name.upper()}")
            ax.set_xlabel("Parameter")
            ax.set_ylabel(metric_name.upper())
            ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('noise_analysis_ldpc_big.png', dpi=150, bbox_inches='tight')
    plt.show()


# -------------------------------
# ORIGINAL IMAGE LOAD SECTION
# -------------------------------

img = Image.open('smalljwphoto.jpg').convert("RGB")
img_array = np.array(img, dtype=int)
original_shape = img_array.shape
original_signal = img_array.flatten()



# -------------------------------
# OPTIONAL: SINGLE ORIGINAL-RUN STYLE
# -------------------------------

# If you still want one single run exactly like before:
single_channel_name = "sinusoidal"   # choose from: awgn, sinusoidal, impulse, multipath, combined
single_param = 0.2                   # for sinusoidal strength

single_result = run_channel_experiment(
    img_array=img_array,
    G=G,
    H=H,
    channel_name=single_channel_name,
    param=single_param,
    sample_rate=8000,
    maxiter=500
)

recovered_array = single_result["recovered_array"]

# Save and display single run result
with open("ldpced_noise.txt", "w") as file:
    json.dump(recovered_array.tolist(), file)

print("Equal arrays:", np.array_equal(img_array, recovered_array))

recovered_array_uint8 = np.clip(recovered_array, 0, 255).astype(np.uint8)
corrupted_img = Image.fromarray(recovered_array_uint8, mode="RGB")
corrupted_img.save("ldpced_output.png")
corrupted_img.show()


# -------------------------------
# PARAMETER SWEEP FOR 16 GRAPHS
# -------------------------------

awgn_params = [1, 2, 3, 4, 5, 6, 7, 8]
sinusoidal_params = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
impulse_params = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
multipath_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

all_results = {}

all_results["awgn"] = sweep_channel(img_array, G, H, "awgn", awgn_params, sample_rate=8000, maxiter=500)
all_results["sinusoidal"] = sweep_channel(img_array, G, H, "sinusoidal", sinusoidal_params, sample_rate=8000, maxiter=500)
all_results["impulse"] = sweep_channel(img_array, G, H, "impulse", impulse_params, sample_rate=8000, maxiter=500)
all_results["multipath"] = sweep_channel(img_array, G, H, "multipath", multipath_params, sample_rate=8000, maxiter=500)

plot_16_graphs(all_results)
