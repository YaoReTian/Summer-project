from Processor import Data
import numpy as np
import noise
import os
import matplotlib.pyplot as plt

def BER(sig, rec):
    sigData = np.array(list(sig),int)
    recData = np.array(list(rec), int)

    totErr = np.sum(sigData!=recData)
    ber = totErr/len(sigData)
    print("Total errors:",totErr)
    print("BER:",totErr/len(sigData))
    return ber



Var = Data()
Var.setDataFromFile("crippling inkperial.jpg")
sigData = Var.toBinaryStr()
signal = Var.bpskMod()

"""
channel = noise.build_channel(
    noise.add_awgn(snr_db=5),
    noise.multipath(delays=[0.001, 0.002], attenuations=[0.5, 0.25]),
    noise.add_impulse_noise(prob=0.5, magnitude=0.1),
    noise.add_sinusoidal_interference(freq=1e10, strength=0.3)
)
"""

# impulse magnitude
impTM = np.linspace(0.1,400,5)
BerTM = []
for i in impTM:

    channel = noise.add_impulse_noise(prob=0.5,magnitude=i)

    signal_after_channel = channel(signal)

    receivedData = Data.bpskDemod(signal_after_channel)

    img = Data()
    img.setDataFromBin(receivedData)
    img.saveFile("received_image.jpg")

    r = img.toBinaryStr()
    k = BER(sigData,r)
    BerTM.append(BER(sigData, r))

print(impTM)
print(BerTM)
fig, ax = plt.subplots()
ax.plot(impTM, BerTM)
plt.show()


