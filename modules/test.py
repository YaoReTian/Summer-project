from Processor import Data
import numpy as np
import noise
import os
import matplotlib.pyplot as plt

Var = Data()
Var.setDataFromFile("crippling inkperial.jpg")
sigData = Var.toBinaryStr()
signal = Var.bpskMod()

def BER(sig, rec):
    sigData = np.array(list(sig),int)
    recData = np.array(list(rec), int)

    totErr = np.sum(sigData!=recData)
    ber = totErr/len(sigData)
    print("Total errors:",totErr)
    print("BER:",totErr/len(sigData))
    return ber

def getBERof(channel):
    signal_after_channel = channel(signal)

    receivedData = Data.bpskDemod(signal_after_channel)

    img = Data()
    img.setDataFromBin(receivedData)

    r = img.toBinaryStr()
    k = BER(sigData,r)

    return k

"""
channel = noise.build_channel(
    noise.add_awgn(snr_db=5),
    noise.multipath(delays=[0.001, 0.002], attenuations=[0.5, 0.25]),
    noise.add_impulse_noise(prob=0.5, magnitude=0.1),
    noise.add_sinusoidal_interference(freq=1e10, strength=0.3)
)
"""

# impulse magnitude
impTM = np.linspace(0.1,400,100)
BerTM = []
for i in impTM:
    channel = noise.add_impulse_noise(prob=0.5,magnitude=i)
    BerTM.append(getBERof(channel))

headTM = "impulse_magnitude, BER"
resTM = np.array([impTM, BerTM]).transpose()
np.savetxt("impulse_magnitude.csv", resTM,delimiter=',',header=headTM, comments='')

# multipath attenuations
attT = np.linspace(0,1,100)
BerAtt = []
for i in attT:
    channel = noise.multipath(delays=[0.001, 0.002], attenuations=[i,i])
    BerAtt.append(getBERof(channel))

headAtt = "multipath_attenuations, BER"
resAtt = np.array([attT, BerAtt]).transpose()
np.savetxt("multipath_attenuations.csv", resAtt,delimiter=',',header=headAtt, comments='')

# awgn
snrT = np.linspace(0.001, 10, 100)
BerSnr = []
for i in snrT:
    channel = noise.add_awgn(i)
    BerSnr.append(getBERof(channel))

headSnr = "awgn_snr, BER"
resSnr = np.array([snrT, BerSnr]).transpose()
np.savetxt("awgn.csv", resSnr,delimiter=',',header=headSnr, comments='')

# sin
sinT = np.linspace(0.1,1000,100)
BerSin = []
for i in sinT:
    channel = noise.add_sinusoidal_interference(freq=1e10, strength=i)
    BerSin.append(getBERof(channel))

headSin = "sinusoidal_strength, BER"
resSin = np.array([impSin, BerSin]).transpose()
np.savetxt("sin_strength.csv", resSin,delimiter=',',header=headSin, comments='')



