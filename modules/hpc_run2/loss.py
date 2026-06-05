import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
from skimage.metrics import structural_similarity as ssim
"""
#soft max function for normalisation 
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

x=np.array([2.0,1.0,0.1])
outputs = softmax(x)
print("Softmax outputs: ",outputs)  

x=torch.tensor([2.0,1.0,0.1])
outputs=torch.softmax(x,dim=0)
print(outputs)

#data sets to test loss on
bce_pred = torch.tensor([0.1,0.7,0.2,0.9,0.12],dtype=torch.float32)
bce_true=torch.tensor([0,1,0,1,0],dtype=torch.float32)


#try from scratch first
def BCE_loss(y_true,y_pred):
    epsilon= 1e-7 #to avoid log(0)
    y_pred = torch.clamp(y_pred,epsilon,1-epsilon) #set boundary to avoid the log 0
    return -torch.mean(y_true*torch.log(y_pred)+(1-y_true)*torch.log(1-y_pred))   
bce_cost = BCE_loss(bce_true, bce_pred)
print("if we use our own loss function: ",bce_cost)

#now use built in
loss_fn=torch.nn.BCELoss()
bce_cost = loss_fn(bce_pred, bce_true)
print("if we use built in loss function: ",bce_cost)

#now add in sigmoid layer with the bce loss to test if we input rare score (logits)

#now prediction tensor spans negative and positive to signify the two possible values

bce_pred = torch.tensor([-20, 20, -3, 15, -12],dtype=torch.float32)
bce_true=torch.tensor([0,1,0,1,0],dtype=torch.float32)
loss_fn=torch.nn.BCEWithLogitsLoss()
bce_cost = loss_fn(bce_pred, bce_true)
print("if we use built in loss function with logits: ",bce_cost)

"""
#signal to noise ratio, snr(noisy channel, original channel)
def snr(x, y): 
    noise = x - y 
    return 10*np.log10(np.mean(x**2) / np.mean(noise**2))



#ber(original, final)
def ber(bits1, bits2):
    bits1 = np.asarray(bits1)
    bits2 = np.asarray(bits2)

    if bits1.shape != bits2.shape:
        raise ValueError("Arrays must have the same shape")

    errors = np.sum(bits1 != bits2)
    return errors / bits1.size

# a = np.array([0, 1, 1, 0, 1])
# b = np.array([0, 0, 1, 1, 1])

# print("BER loss",ber(a, b))  # 0.4
def stuctural_sim(img1, img2):
    return ssim(img1, img2, channel_axis=2,data_range=255)

def mse(x, y): 
    return np.mean((x - y)**2) 

#peak signal to noise ratio 
def PSNR(mse): 
    return 10 * np.log10(255**2 /mse)
