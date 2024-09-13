import numpy as np
import pandas as pd
import torch
from torch.nn.functional import binary_cross_entropy, cross_entropy
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

## Common utility for batches generation from train and test data 


## Common utility for initializing loss function basis input argument


def CustomizedLoss(outputs,norm_inputs,wt,mean,var,log_var,batch_size):
    #print("weight is"+str(wt))
    regen_err= binary_cross_entropy(outputs,norm_inputs,reduction='mean')
    #regen_err= cross_entropy(outputs,norm_inputs,reduction='mean')
    one_tensor=torch.ones_like(var)
    KL_div= -0.5*torch.sum(1+log_var-torch.pow(mean,2)-torch.exp(log_var))/batch_size
    #print("regen_error is")
    #print(regen_err)
    #print("KLError is")
    #print(KL_div)
    loss=regen_err+wt*KL_div ##This part can have weight decay for the KL_div loss
    return loss,regen_err,wt*KL_div


def show_image(x):
    print(x.size())
    x = x.view(184,3, 32, 32)
    fig = plt.figure()
    img_orig=x[0].detach().numpy().T
    print(img_orig.shape)
    #img = np.moveaxis(img_orig, 0, -1)
    plt.imshow(img_orig)
    plt.imsave("sample_img.png",img_orig)


class CustomData(Dataset):
    def __init__(self,data):
        self.data=data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]
    
    def custom_collate(batch):
        img=[item[0] for item in batch]
        label=[item[1] for item in batch]
        return torch.tensor(img),torch.tensor(label)