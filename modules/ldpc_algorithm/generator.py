import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message 
import json

# -----------------------------
# 1. Build an LDPC code - use this to encode all messages 
# this is high protection ldpc with rate of 0.5625
#    We want total codeword length = 2^m, choose 2048 in this case
# -----------------------------
n = 2048      # codeword length
d_v = 9      # number of parity-check connections per variable node
d_c = 16        # number of parity-check connections per check node
seed = 42       # fix the randomness of the parity bits connections



H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True, seed=seed)   
# Check the message length supported by this generator matrix
k = G.shape[1]
print("Generator matrix shape:", G.shape)
print("Expected message length k:", k)

with open("generator.txt","w") as file:
    json.dump(G.tolist(), file)

with open("hamming_matrix.txt","w") as file:
    json.dump(H.tolist(),file)
    
