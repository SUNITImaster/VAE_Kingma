

import torch.nn as nn
import numpy as np

class config_VAE_Kingma_v1:
    def __init__(self,LossFn_version,dataset,Arch_type,Decay_option,seed):
        # output config
        self.VAE_logic="Basic_VAE_Kingma"
        seed_str = "seed_" + str(seed)
        self.output_path = "results/{}-{}-{}-{}/".format(
            dataset,Arch_type,Decay_option,seed_str
        )
        self.model_output = self.output_path + "modelweights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 10
        self.summary_freq = 1

        # model and training config
        self.num_batches = 100  # number of batches trained on
        self.batch_size = 256  # number of steps used to compute each policy update
        self.learning_rate = 5e-3
    
        self.decay=np.where(Decay_option=="WithDecay",True,False)
        self.num_epochs=51
        
        if(self.decay==True):
            self.wt=[0.01,0.01,0.01,0.01]
        else:
            self.wt=[0.01]


"""
            self.Encoder = nn.Sequential(OrderedDict([
                            ('Conv1',nn.Conv2d(in_channels=3, out_channels=256,kernel_size=(3,3),stride=2,padding=1)),
                            ('Relu1',nn.ReLU()),
                            ('Conv2',nn.Conv2d(in_channels=256, out_channels=512,kernel_size=(3,3),stride=2,padding=1)),
                            ('Relu2',nn.ReLU()),
                            ('Conv3',nn.Conv2d(in_channels=512, out_channels=32,kernel_size=(2,2),stride=2,padding=0)),
                            ('Flat4',nn.Flatten())
                            ##output now is len 16 vector
                            ]))
            self.Encoder_mean=nn.Sequential(OrderedDict([('Linear2_1',nn.Linear(256,256)),
                                            ('Tanh2_1',nn.Tanh())]))
            self.Encoder_var=nn.Sequential(OrderedDict([('Linear2_2',nn.Linear(256,256)),
                                            ('Sig2_2',nn.Sigmoid())]))
            
            self.Decoder=nn.Sequential(OrderedDict([
                            ('LinearD_0',nn.Linear(256,256)),
                            ('LinearD_1',nn.Linear(256,256)),
                            ('ReluD_1',nn.ReLU()),
                            ('UnFlatten0',nn.Unflatten(1,(16,4,4))),
                            #No relu as We don't want to loose signal here
                            ('Deconv1',nn.ConvTranspose2d(in_channels=16, out_channels=64,kernel_size=(3,3),stride=1)),
                            #('ReluD_2',nn.ReLU()),
                            ('Upsample1',nn.UpsamplingNearest2d(scale_factor=2)),#input:6x6, output:12x12
                            ('Deconv2',nn.ConvTranspose2d(in_channels=64, out_channels=128,kernel_size=(3,3),stride=1)),
                            #('ReluD_3',nn.ReLU()),
                            ('Upsample2',nn.UpsamplingNearest2d(scale_factor=2)),#input:14x14,output:28x28
                            ('Deconv3',nn.ConvTranspose2d(in_channels=128, out_channels=3,kernel_size=(5,5),stride=1)),
                            ('Sig1',nn.Sigmoid())
                            ]))

"""




def get_config(LossFn_version, dataset,Arch_type, Decay_option, seed):
    
    return config_VAE_Kingma_v1(LossFn_version, dataset,Arch_type, Decay_option, seed )
    
       