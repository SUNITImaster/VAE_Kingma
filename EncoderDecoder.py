import torch
import torch.nn as nn
from Utility import CustomizedLoss,show_image,CustomData
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader, random_split,RandomSampler
from torchvision.datasets import MNIST,CIFAR10,CIFAR100
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import cv2
import numpy as np


class EncoderDecoder(nn.Module):

    def __init__(self,LossFn_version,Arch_type,dataset,Decay_option,seed,config):
        super(EncoderDecoder, self).__init__()
        PATH="D:/Suniti/GitPythonRepo/VAE_Kingma/results/CIFAR10-WithConv-NoDecay-seed_4/modelweights/model_optim_epoch_20.pt"
        self.config=config
        ##Initializing the train,test and val batch data generator 
        train_datagen,test_datagen,val_datagen=self.batch_data(dataset)
        ##tested that batch data has good mix of all 10 image types.
        EncoderDecoder.define_network(self,Arch_type)
        #loaded_dict=torch.load(PATH,weights_only=True)
        #EncoderDecoder.load_state_dict(self,state_dict=loaded_dict["model_dict"])
        
        outputs,norm_inputs=EncoderDecoder.train_network(self,train_datagen,test_datagen,val_datagen)
        #show_image(norm_inputs)
        show_image(outputs)

        ##Nnet architecture wil depend upon input/output image size and the size of Z layer. 
        ## Will also depend on no. of layers and Type of Layers (FC vs. Conv)

    def define_network(self,Arch_type):
        if(Arch_type=="OnlyFC"):
            ## only FC layers
            self.z_dim = 16
            self.Encoder = nn.Sequential(OrderedDict([
                            ('Linear0',nn.Linear(self.en_input_dim,512)),
                            ('Reul0',nn.Relu()),
                            ('Linear1',nn.Linear(512,256)),
                            ('Reul1',nn.Relu())
                            ]))
            self.Encoder_mean=nn.Sequential(OrderedDict([('Linear2_1',nn.Linear(256,z_dim)),
                                            'Sig2_1',nn.Tanh()]))
            self.Encoder_std=nn.Sequential(OrderedDict([('Linear2_2',nn.Linear(256,z_dim)),
                                            'Sig2_2',nn.Sigmoid()]))
            
            self.Decoder=nn.Sequential(OrderedDict([
                            ('Linear0',nn.Linear(z_dim,256)),
                                ('Reul0',nn.Relu()),
                                ('Linear1',nn.Linear(256,512)),
                            ('Sigm1',nn.Sigmoid())
                            ('Linear_final',nn.Linear(512,self.en_input_dim))]))

        elif(Arch_type=="WithConv"):

            ## with Conv & Decon layers 
            z_dim = 256
            self.Encoder = nn.Sequential(OrderedDict([
                            ('Conv1',nn.Conv2d(in_channels=3, out_channels=256,kernel_size=(3,3),stride=2,padding=1)),
                            ('Batchnorm1',nn.BatchNorm2d(256)),
                            ('Relu1',nn.ReLU()),
                            ('Conv2',nn.Conv2d(in_channels=256, out_channels=512,kernel_size=(3,3),stride=2,padding=1)),
                            ('Batchnorm2',nn.BatchNorm2d(512)),
                            ('Relu2',nn.ReLU()),
                            ('Conv3',nn.Conv2d(in_channels=512, out_channels=16,kernel_size=(2,2),stride=2,padding=0)),
                            ('Batchnorm3',nn.BatchNorm2d(16)),
                            ('Relu3',nn.ReLU()),
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

        else:
            print("no valid Encoder Decoder Architecture provided so exiting")
            print(exit)
        
        
    def forward(self,x):
        x_enc=self.Encoder(x)
        mean=self.Encoder_mean(x_enc)
        #var=torch.ones_like(mean)
        var=self.Encoder_var(x_enc) ##this var is the std deviation of noise variable
        epsilon = torch.randn_like(var)        # sampling random noise  
           
        z = mean + var*epsilon 
        #z_inv= 0.5*(torch.log(1+z)-torch.log(1-z)) #taking the tanh inverse of z 
        #print("shape of sampled z vector is")
        #print(z.size())
        #print(z_reshaped.size())
        X_hat=self.Decoder(z)
        #print("shape of X_hat is")
        #print(X_hat.size())
        return X_hat,mean,var

    
    def batch_data(self,x):
        rsampler=RandomSampler(x)
        train_data,test_data,val_data=random_split(x,lengths=[35000,10000,5000])
        train_datagen = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        test_datagen = DataLoader(test_data, batch_size=self.config.batch_size, shuffle=True)
        val_datagen = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=True)
        return train_datagen,test_datagen,val_datagen
    
    def train_network(self,train_datagen,test_datagen,val_datagen):

         ## if Weight Decay is given , then loss function has to change after every 10 epochs say.
        
        optimizer=optim.Adam(self.parameters(),lr=self.config.learning_rate,betas=(0.9025,0.95))
        scheduler=lr_scheduler.LinearLR(optimizer,start_factor=1,end_factor=0.1,total_iters=45)
        #optimizer=optim.RMSprop(self.parameters(),lr=self.config.learning_rate,alpha=0.99)
        idx=0
        for epoch in range(self.config.num_epochs):  # loop over the dataset multiple times
        ##if epoch in range 0-10,10-20, so on, use corresponding weight decay for loss function.
            running_loss = 0.0
            regen_loss=0.0
            KL_loss=0.0

            if (self.config.decay==True):
                idx=np.int8(np.floor(epoch/25)) # This ensures 4 different weight indices
            else:
                idx=0
            for batch_idx, data in enumerate(train_datagen):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                ##checking below if all type of images are being sampled or not
                #print(f"Feature batch shape: {inputs.size()}")
                #print(f"Labels batch shape: {labels.size()}")
                #print(torch.bincount(labels))
                #print(torch.unique(labels))
                #print(inputs[0].size())
                #print(inputs[100])
                #img = (inputs[100].squeeze().transpose(0,1)).transpose(1,2)
                #plt.imshow(img)
                #plt.show()
                #break
                norm_inputs=torch.from_numpy(cv2.normalize(np.asarray(inputs),None,0,1,cv2.NORM_MINMAX))
                #print(norm_inputs[0])
                # zero the parameter gradients
                optimizer.zero_grad()

                

                # forward + backward + optimize
                outputs,mean,var = self.forward(norm_inputs)
                #print("shape of input batch is")
                #print(norm_inputs.size())
                #print("shape of output image batch is")
                loss,regen_err,KL_div = CustomizedLoss(outputs,norm_inputs,self.config.wt[idx],mean,var,torch.log(var),self.config.batch_size)
                running_loss=running_loss+loss.item()
                regen_loss=regen_loss+regen_err.item()
                KL_loss=KL_loss+KL_div.item()
                loss.backward()
                #EncoderDecoder.print_grad(self)#this function will print gradient of the intermediate layer specified 
                optimizer.step()

            #print(outputs[0])
            print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss per batch: ", running_loss / (batch_idx))
            print("\tRegenAvgLoss is ",regen_loss/batch_idx,"\tand KLAvgLoss is ",KL_loss/batch_idx)
            print("\tLearning rate is",optimizer.param_groups[0]["lr"])
            scheduler.step()
            if(epoch%self.config.record_freq==0):
                filenamepath=self.config.model_output+"model_optim_epoch_"+str(epoch)+".pt"
                torch.save(
                    {'epoch':epoch,
                     'model_dict':self.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'kl_loss':KL_loss
                     }, filenamepath
                )
        
        return outputs,norm_inputs



    def print_grad(self):
        for name,module in self.named_children():
            if name in ['Conv3', 'Deconv3','Sig1']:
                for p in module.parameters():
                    print(f"<{name}>:{p.grad}")


    def print_filters(self,Layer,nrows=8):
        n,c,w,h=Layer.shape()
        ncol=np.floor(n//nrows)
        grid=make_grid(tensor=Layer,nrow=nrows,normalize=True)
        plt.figure(figsize=(nrows,ncol))
        plt.imshow(grid.numpy().transpose(1,2,0))
