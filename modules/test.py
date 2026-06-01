from Processor import Data
import numpy as np
import noise
import os

"""
import os
for f in os.listdir('.'):
    if f.endswith('.jpg'):
        print(f)

print(os.listdir('.'))
print(os.listdir('modules'))
print(os.getcwd())
"""
time = np.linspace(0,1e-7,20)

Var = Data()
Var.setDataFromFile("crippling inkperial.png")
signal = Var.bpskMod()
print(signal)

channel = noise.build_channel(
    noise.add_awgn(snr_db=5),
    noise.multipath(delays=[0.001, 0.002], attenuations=[0.5, 0.25]),
    noise.add_impulse_noise(prob=0.8, magnitude=400),
    noise.add_sinusoidal_interference(freq=1e10, strength=0.3)
)

signal_after_channel = channel(signal)

receivedData = Data.bpskDemod(signal_after_channel)

print("Data before:")
print(signal(time))
print("Sum:", np.sum(signal(time)))

print("Data after:")
print(signal_after_channel(time))
print("Sum:", np.sum(signal_after_channel(time)))


img = Data()
img.setDataFromBin(receivedData)
img.saveFile("received_image.png")

r = Var.toBinaryStr()
err = np.logical_xor(r,img.toBinaryStr())
print(err)
totErr = np.sum(err)
print("Total errors:",totErr)
print("BER:",totErr/len(r))
