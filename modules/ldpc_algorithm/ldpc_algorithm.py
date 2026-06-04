import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message 
import time
import json 


#start clock
start=time.time()

#initialize some input parameters
bits=np.array([1,0,1,1,0,1,0,1,0,0]*200,dtype=float)
snr=5

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

def encode_block(block, G,snr):
    """
    Encodes one message block using generator matrix G over     
    """
    block = np.asarray(block, dtype=float).ravel()
    G = np.asarray(G, dtype=float)

    
    codeword = encode(G, block, snr)
    return np.asarray(codeword, dtype=float).ravel()

def encode_bit_array_ldpc(bits, G,snr, k=904, return_blocks=False):
    """
    Encode a bit array of any length using LDPC generator matrix G
    """
    n, _ = G.shape
    blocks, original_len, pad_len = split_into_blocks(bits, k=k)

    encoded_blocks = np.empty((blocks.shape[0], n), dtype=float)

    for i in range(blocks.shape[0]):
        encoded_blocks[i] = encode_block(blocks[i], G,snr)

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

result = encode_bit_array_ldpc(bits, G,snr, k=904, return_blocks=True)

print("Original length:", result["original_length"])
print("Pad length:", result["pad_length"])
print("Number of blocks:", result["num_blocks"])
print("Message block size:", result["message_block_size"])
print("Codeword length:", result["codeword_length"])
print("Encoded bitstream length:", len(result["encoded_bits"]))
print("Encoded blocks shape:", result["encoded_blocks"].shape)
print("Encoded blocks",result["encoded_bits"])

#check time
end=time.time()
length= end-start
print(f"It took {length} time to compute")


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
def decode_ldpc_blocks(channel_values, H, G, original_length,codeword_length, snr, k_expected=904, maxiter=1000):
    """
    Decode a stream of LDPC channel observations block-by-block.
    """

    codeword_blocks = split_codewords(channel_values, codeword_length)

    decoded_blocks = []

    for i, y in enumerate(codeword_blocks):
        d = decode(H, y, snr,maxiter=maxiter)
        print(d)
        msg = d[:k_expected]
        msg = np.asarray(msg, dtype=np.float32).ravel()
        decoded_blocks.append(msg)

    decoded_bits = np.concatenate(decoded_blocks)
    decoded_bits = decoded_bits[:original_length]

    return decoded_bits

decoded_bits = decode_ldpc_blocks(
    channel_values=result["encoded_blocks"],
    H=H,
    G=G,
    original_length=result["original_length"],
    codeword_length=result["codeword_length"],
    snr=5,
    k_expected=904,
    maxiter=200
)

