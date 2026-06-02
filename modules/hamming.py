import numpy as np

class Hamming:
    _gT = np.matrix([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    _h = np.matrix([
        [1, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1],
        [0, 0, 1, 1, 1, 0, 1]
    ])

G = np.array(Hamming._gT.T)   # (4x7) generator matrix
H = np.array(Hamming._h)       # (3x7) parity-check matrix

# Recovery matrix — extracts the 4 data bits from positions 3,4,5,6
R = np.array([
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]
])

# encode
d = np.array([1, 0, 1, 1])                        # original 4-bit message
c = (np.dot(d, G) % 2).astype(int)                # encoded 7-bit codeword
print("Original message:  ", d.tolist())
print("Encoded codeword:  ", c.tolist())

# 1 bit error introduced
c_received = c.copy()
c_received[2] ^= 1                                # flip bit at position 2
print("Received codeword: ", c_received.tolist())

# syndrome calculation
s = (np.dot(H, c_received.T) % 2).astype(int)
print("Syndrome:          ", s.tolist())

# error detection and correction
if s.tolist() == [0, 0, 0]:
    print("No errors detected.")
    corrected = c_received
else:
    # Convert syndrome from binary to integer to get error position
    # s[0] is LSB, s[2] is MSB  →  position = s[0]*1 + s[1]*2 + s[2]*4
    error_pos = int(s[0] * 1 + s[1] * 2 + s[2] * 4) - 1  # subtract 1 for 0-indexing
    print(f"Error detected at bit position: {error_pos} (0-indexed)")

    corrected = c_received.copy()
    corrected[error_pos-1] ^= 1                     # flip the erroneous bit back
    print("Corrected codeword:", corrected.tolist())

# recover original message
recovered_message = (np.dot(R, corrected.T) % 2).astype(int)
print("Recovered message: ", recovered_message.tolist())
print("Match:", np.array_equal(d, recovered_message))