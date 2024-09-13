# -*- coding: UTF-8 -*-
"""
1. This is the main code which takes takes as input the following parameters at runtime : 
a.LossFunction
b.EncoderDecoder Architecture
c. Decaying weights schedule for KL divergence.

2. Add code to -
a. Save Encoder and Decoder architecture at few checkpoints. Saved Encoder can then be used for GAN Generator.

3. Debugging :
a. As of now , the VAE is barely able to reduce the reconstruction loss. This could be due to Poor Encoder, Decoder architecture.
b. Print out Encoded features from the Encoder to see if it is learning correctly.
"""

import argparse
import pandas as pd
import numpy as np
import os
import torch
from config import get_config
import time
import random
import sklearn
import pdb
import logging
from EncoderDecoder import EncoderDecoder
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST,CIFAR10,CIFAR100
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument(
    "--LossFn_version", required=True, type=str, choices=["RegenLoss", "BinaryCE"]
)
parser.add_argument("--Arch_type", dest="Arch_type", choices=["WithConv", "OnlyFC"])
parser.add_argument("--dataset", dest="dataset", choices=["MNIST", "CIFAR10","CIFAR100"])
parser.add_argument("--Decay_option", dest="Decay_option", choices=["WithDecay", "NoDecay"])
parser.add_argument("--seed", type=int, default=1)



if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # initialize the code configuration parameters.
    config = get_config(args.LossFn_version, args.dataset,args.Arch_type, args.Decay_option, args.seed )
    # Load Train test val datasets as per args
    if(args.dataset=="CIFAR10"):
        vision_dataset = CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
        )
    print(len(vision_dataset))

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
        os.makedirs(config.model_output)
        
    """
    for i, data in enumerate(train_datagen, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            print(f"Feature batch shape: {inputs.size()}")
            print(f"Labels batch shape: {labels.size()}")
            print(inputs[0].size())
            img = (inputs[0].squeeze().transpose(0,1)).transpose(1,2)
            label = inputs[0]
            plt.imshow(img)
            plt.show()
            break
            
    """
    #train model
    model = EncoderDecoder(args.LossFn_version,args.Arch_type,vision_dataset,args.Decay_option,args.seed,config)
