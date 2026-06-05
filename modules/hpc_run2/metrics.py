import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import loss
import ast 

"""Trial"""
# #reading the two files to compare
# data=str(input("what data file do you want to read:")+".txt")
# with open(data, "r") as file:
#     corrupted_list = ast.literal_eval(file.read())
# corrupted_arr = np.array(corrupted_list,dtype=np.uint8)
# img_arr = np.array(Image.open('smalljwphoto.jpg').convert("RGB"),dtype=np.uint8)# convert to RGB to ensure we have 3 channels
# file.close()
# print(corrupted_arr.shape,img_arr.shape)
"""End Trial"""
#conversions
#converts a 1D array of integers to a 1D array of bits
def ints_to_flat_bits(arr, width=8):
    return [(int(n) >> i) & 1 for n in arr for i in range(width - 1, -1, -1)]




#metric pipeline 

"""
There are two main files this is used on 
1. noisechannel.py -no ldpc
ber: img_bits, corrupted_bits
ssim: img_array, recovered_array
snr: flat_signal, received_signal
psnr: original_signal, recovered_pixels

2. noisepipeline.py -ldpc
ber: binary_arr, recovered_bits
ssim: img_array, recovered_array
snr: clean_soft_signal, soft_signal
psnr: original_signal, recovered_pixels

"""

#params declared as above, 8 params but psnr takes flattened ssim arrays
def metric_pipeline(img_arr,corrupted_arr,trans,receive):
    img_bits = ints_to_flat_bits(np.array(img_arr).flatten())
    corrupted_bits = ints_to_flat_bits(np.array(corrupted_arr).flatten())
    ber = loss.ber(img_bits, corrupted_bits)
    ssim_score = loss.stuctural_sim(img_arr,corrupted_arr)
    snr = loss.snr(trans,receive)
    psnr = loss.PSNR(loss.mse(np.array(img_arr).flatten(),np.array(corrupted_arr).flatten()))
    return ber,ssim_score, snr, psnr

#examplar usage
# ber,ssim_score,snr,psnr = metric_pipeline(img_arr,corrupted_arr,trans,received)
# print(f"ber:{ber}, ssim:{ssim_score}, snr:{snr}, psnr:{psnr}")


