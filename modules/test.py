from Processor import Data 
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


Var = Data().setDataFromFile("C:/Users/User/OneDrive - Imperial College London/Summer-project/modules/cutout_hlsp_appp_hst_wfpc2_sfd-pu4k2ho01_f606w_v2_sci_sci.jpg")
signal = Var.bpskMod()

channel = noise.build_channel(
    noise.add_awgn(snr_db=20),
    noise.multipath(delays=[0.001, 0.002], attenuations=[0.5, 0.25]),
    noise.add_impulse_noise(prob=0.001, magnitude=5),
    noise.add_sinusoidal_interference(freq=1000, strength=0.3)
)

signal_after_channel = channel(signal)

receivedData = Data.bpskDemod(signal_after_channel)

img = Data().setDataFromBin(receivedData)
img.saveFile("C:/Users/User/OneDrive - Imperial College London/Summer-project/modules/received_image.jpg")
