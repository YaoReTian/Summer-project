import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim_metric
import noise  # your noise module

# ── Load & prepare image ──────────────────────────────────────────────────────
img = Image.open('data/rick.jpg').convert("RGB")
img_array = np.array(img, dtype=np.float32)
original_shape = img_array.shape

flat_signal = img_array.flatten()
flat_signal = (flat_signal / 127.5) - 1.0
N = len(flat_signal)

# ── Waveform factory (from pipeline) ─────────────────────────────────────────
def make_waveform(signal, sample_rate=8000):
    Ts = 1 / sample_rate
    N = len(signal)
    def waveform(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
        indices = np.floor(t_arr / Ts).astype(int) % N
        indices = np.clip(indices, 0, N - 1)
        return signal[indices]
    return waveform, Ts, N

sample_rate = 8000
waveform, Ts, N = make_waveform(flat_signal, sample_rate=sample_rate)
t = np.linspace(0, N * Ts, N, endpoint=False)

# ── Metric functions (from analysis module) ───────────────────────────────────
def calc_snr(original, received):
    noise_sig = received - original
    noise_power = np.mean(noise_sig**2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(np.mean(original**2) / noise_power)  # dB

def calc_mse(original, received):
    # normalise to [0,1] for a scale-independent measure
    o = (original + 1.0) / 2.0
    r = (received + 1.0) / 2.0
    return np.mean((o - r)**2)

def calc_psnr(original, received):
    mse_val = calc_mse(original, received)
    if mse_val == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse_val)  # peak = 1.0 since signal in [0,1]

def calc_ssim(original, received, shape):
    # denormalise to [0, 255] uint8 for skimage
    o_img = ((np.clip(original, -1, 1) + 1.0) * 127.5).astype(np.uint8).reshape(shape)
    r_img = ((np.clip(received, -1, 1) + 1.0) * 127.5).astype(np.uint8).reshape(shape)
    return ssim_metric(o_img, r_img, channel_axis=2, data_range=255)

def calc_ber(original, received):
    # quantise both signals to 8-bit and compare bit-by-bit
    o_uint8 = ((np.clip(original, -1, 1) + 1.0) * 127.5).astype(np.uint8)
    r_uint8 = ((np.clip(received, -1, 1) + 1.0) * 127.5).astype(np.uint8)
    # unpack each byte into 8 bits
    o_bits = np.unpackbits(o_uint8)
    r_bits = np.unpackbits(r_uint8)
    return np.sum(o_bits != r_bits) / o_bits.size

def apply_and_measure(channel_fn):
    """Apply a channel, sample the output, return all four metrics."""
    noisy_wf = channel_fn(waveform)
    received = noisy_wf(t)
    snr_val  = calc_snr(flat_signal, received)
    psnr_val = calc_psnr(flat_signal, received)
    ssim_val = calc_ssim(flat_signal, received, original_shape)
    ber_val  = calc_ber(flat_signal, received)
    return snr_val, psnr_val, ssim_val, ber_val

# ── Parameter sweeps ──────────────────────────────────────────────────────────

# 1. Gaussian AWGN  — SNR_dB from 4 to 10 (step 1)
awgn_snrs   = np.arange(4, 11, 1)          # [4,5,6,7,8,9,10]
awgn_results = [apply_and_measure(
                    noise.build_channel(noise.add_awgn(snr_db=s)))
                for s in awgn_snrs]

# 2. Multipath — attenuation from 0.1 to 0.7 (step 0.1), both paths equal
mp_attens   = np.round(np.arange(0.1, 0.71, 0.1), 2)   # [0.1 … 0.7]
mp_results  = [apply_and_measure(
                    noise.build_channel(
                        noise.multipath(delays=[0.001, 0.002],
                                        attenuations=[a, a])))
               for a in mp_attens]

# 3. Sinusoidal interference — freq from 1000 to 7000 (step 1000), strength=0.1
sin_freqs   = np.arange(1000, 8000, 1000)  # [1000 … 7000]
sin_results = [apply_and_measure(
                    noise.build_channel(
                        noise.add_sinusoidal_interference(freq=f, strength=0.1)))
               for f in sin_freqs]

# 4. Impulse noise — prob from 0.001 to 0.007 (step 0.001), magnitude=0.3
imp_probs   = np.round(np.arange(0.001, 0.008, 0.001), 3)
imp_results = [apply_and_measure(
                    noise.build_channel(
                        noise.add_impulse_noise(prob=p, magnitude=0.3)))
               for p in imp_probs]

# ── Unpack results ────────────────────────────────────────────────────────────
def unpack(results):
    snrs, psnrs, ssims, bers = zip(*results)
    return (np.array(snrs), np.array(psnrs),
            np.array(ssims), np.array(bers))

awgn_snr_v, awgn_psnr_v, awgn_ssim_v, awgn_ber_v = unpack(awgn_results)
mp_snr_v,   mp_psnr_v,   mp_ssim_v,   mp_ber_v   = unpack(mp_results)
sin_snr_v,  sin_psnr_v,  sin_ssim_v,  sin_ber_v  = unpack(sin_results)
imp_snr_v,  imp_psnr_v,  imp_ssim_v,  imp_ber_v  = unpack(imp_results)

# ── Plotting ──────────────────────────────────────────────────────────────────
METRIC_LABELS = ['SNR (dB)', 'PSNR (dB)', 'SSIM', 'BER']

noise_configs = [
    {
        'label'   : 'Gaussian AWGN',
        'x'       : awgn_snrs,
        'xlabel'  : 'Input SNR (dB)',
        'metrics' : [awgn_snr_v, awgn_psnr_v, awgn_ssim_v, awgn_ber_v],
        'color'   : 'steelblue',
    },
    {
        'label'   : 'Multipath',
        'x'       : mp_attens,
        'xlabel'  : 'Attenuation per Path',
        'metrics' : [mp_snr_v, mp_psnr_v, mp_ssim_v, mp_ber_v],
        'color'   : 'darkorange',
    },
    {
        'label'   : 'Sinusoidal Interference  (strength=0.1)',
        'x'       : sin_freqs,
        'xlabel'  : 'Interference Frequency (Hz)',
        'metrics' : [sin_snr_v, sin_psnr_v, sin_ssim_v, sin_ber_v],
        'color'   : 'seagreen',
    },
    {
        'label'   : 'Impulse Noise  (magnitude=0.3)',
        'x'       : imp_probs,
        'xlabel'  : 'Impulse Probability',
        'metrics' : [imp_snr_v, imp_psnr_v, imp_ssim_v, imp_ber_v],
        'color'   : 'crimson',
    },
]

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.suptitle('Channel Noise Analysis — Effect on Image Quality Metrics',
             fontsize=16, fontweight='bold', y=1.01)

for col, cfg in enumerate(noise_configs):
    for row, (metric_vals, metric_label) in enumerate(
            zip(cfg['metrics'], METRIC_LABELS)):

        ax = axes[row][col]
        ax.plot(cfg['x'], metric_vals,
                marker='o', linewidth=2,
                color=cfg['color'], markersize=5)
        ax.set_title(f"{cfg['label']}", fontsize=9, fontweight='bold')
        ax.set_xlabel(cfg['xlabel'], fontsize=8)
        ax.set_ylabel(metric_label, fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=7)

        # tidy x-axis tick density
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))

plt.tight_layout()
plt.savefig('data/noise_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Analysis complete — saved to data/noise_analysis.png")
