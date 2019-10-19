import numpy as np
import torch
import sys,os

################################ INPUT #######################################

maps = 4096
seed = 6    #random seed
f_best_model = 'BestModel_UNET_New.pt'
##############################################################################

# Create containers for HI True maps, DM (input) maps, and HI Estimated (Analytical Input) Maps

HI_True = np.zeros((4096, 32, 32), dtype = np.float32)
HI_esti = np.zeros((4096, 32, 32), dtype = np.float32)
DM_True = np.zeros((4096, 32, 32), dtype = np.float32)

count = -1
# Load sliced maps into containers
for i in range(64):
    for j in range(64):
        count += 1
        HI_True[count] = np.load('/home/helen/HI_Maps_Sliced/HI_map_%d_%d.npy'%(i,j))
        DM_True[count] = np.load('/home/helen/DM_Maps_Sliced/DM_map_%d_%d.npy'%(i,j))
        HI_esti[count] = np.load('/home/helen/Esti_Sliced/HI_esti_%d_%d.npy'%(i,j))
     
###################################################################
# This function implements the transformation we do to the data
def transform(x):
    return(np.log(x+1.0)) #x**0.25
##################################################################

# Take the log (transform)
DM_True = transform(DM_True)
HI_True = transform(HI_True)

######## Normalize Data #######

# Mean and std values
DM_True_mean, DM_True_std = np.mean(DM_True), np.std(DM_True)
HI_True_mean, HI_True_std = np.mean(HI_True), np.std(HI_True)

# Normalize Matrices
DM_True = (DM_True - DM_True_mean)/DM_True_std
HI_True = (HI_True - HI_True_mean)/HI_True_std

HI_min = np.min([HI_True, HI_esti])
HI_max = np.max([HI_True, HI_esti])

# Convert to torch tensor
DM_True = torch.tensor(DM_True, dtype=torch.float)
HI_True = torch.tensor(HI_True, dtype=torch.float)
HI_esti = torch.tensor(HI_esti, dtype=torch.float)
    
import numpy as np
import torch
import torchvision
import torch.nn as nn

############################# Create UNet Model ################################
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,  out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True)
    )   

def double_conv_final(in_channels, out_channels1, out_channels2, out_channels3):
    return nn.Sequential(
        nn.Conv2d(in_channels,   out_channels1, 1, padding=0),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels1, out_channels2, 1, padding=0),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels2, out_channels3, 1, padding=0),
        nn.LeakyReLU(inplace=True)
    )

class UNet(nn.Module):

    def __init__(self):
        super(UNet,self).__init__()
                
        self.dconv_down1 = double_conv(2, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        self.dconv_down4 = double_conv(64, 128)        

        self.maxpool = nn.MaxPool2d(2)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', 
                                    align_corners=True)        
        
        self.dconv_up3 = double_conv(64 + 128,    64)
        self.dconv_up2 = double_conv(32 + 64,     32)
        self.dconv_up1 = double_conv(16 + 32 + 2, 16)
        self.conv_last = nn.Conv2d(16, 1, 1)    
        #self.conv_last = double_conv_final(18,8,4,1)

        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x1    = self.maxpool(conv1)
        
        conv2 = self.dconv_down2(x1)
        x2    = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x2)
        x3    = self.maxpool(conv3)   
        
        x4 = self.dconv_down4(x3)
        
        x5 = self.upsample(x4)  
        x6 = torch.cat([x5, conv3], dim=1)
        x7 = self.dconv_up3(x6)

        x8 = self.upsample(x7)        
        x9 = torch.cat([x8, conv2], dim=1)       
        x10 = self.dconv_up2(x9)

        x11 = self.upsample(x10)        
        x12 = torch.cat([x11, conv1, x], dim=1)           
        x13 = self.dconv_up1(x12)
        out = self.conv_last(x13)
 
        #x14 = torch.cat([x13, x], dim=1)  
        #out = self.conv_last(x14) 
        
        return out
        
        
########### MAIN LOOP ##########
model = UNet()
model.load_state_dict(torch.load('BestModel_UNET_New.pt'))
model.eval() 

HI_Predicted = np.zeros((4096, 32, 32), dtype = np.float32)

for i in range(4096): 
    Map_in = torch.cat([DM_True[i].unsqueeze(0), 
                         HI_esti[i].unsqueeze(0)], dim=0).unsqueeze(0)
    HI_pred = model(Map_in)[0,0,:,:].detach().numpy() 

    DM = DM_True[i].numpy()
    DM_min,DM_max,DM_mean,DM_std = np.min(DM),np.max(DM),np.mean(DM),np.std(DM)
    HI0 = HI_True[i].numpy()
    HI1 = HI_esti[i].numpy()

    HI_min = np.min([HI0, HI1, HI_pred])
    HI_max = np.max([HI0, HI1, HI_pred])
    
    HI_Predicted[i] = HI_pred
    
    loss = np.mean((HI_pred - HI0)**2)
    print('loss = %.3f'%(loss))
   
   
######################### Full Map ##########################
HI_Pred_Full = np.zeros((2048, 2048), dtype = np.float32)
count = -1

#Piece together the 32x32 maps 
for i in range(64):
    for j in range(64):
        count += 1
        HI_Pred_Full[32*i:32+(32*i), (32*j):32*(j+1)] = HI_Predicted[count]


####################### Plot Maps ##########################
#Load Full Maps
HI_Full_True = np.load('HI_new_full_map_1.npy').astype('float32')
DM_Full_True = np.load('DM_new_full_map_1.npy').astype('float32')

DM_Full_True = transform(DM_Full_True)
HI_Full_True = transform(HI_Full_True)

######## Normalize Data #######

# Mean and std values
DM_Full_True_mean, DM_Full_True_std = np.mean(DM_Full_True), np.std(DM_Full_True)
HI_Full_True_mean, HI_Full_True_std = np.mean(HI_Full_True), np.std(HI_Full_True)


# Normalize Matrices
DM_Full_True = (DM_Full_True - DM_Full_True_mean)/DM_Full_True_std
HI_Full_True = (HI_Full_True - HI_Full_True_mean)/HI_Full_True_std

#Create Plots and Save as PDF
f = plt.figure()

plt.subplot(421)
plt.imshow(DM_Full_True)
plt.colorbar()
plt.title('True Map of Dark Matter')

plt.subplot(422)
plt.imshow(HI_Full_True)
plt.colorbar()
plt.title('True Map of HI')

plt.subplot(423)
HI_estimated_full = np.load('HI_estimated_full.npy').astype('float32')
plt.imshow(HI_estimated_full)
plt.colorbar()
plt.title('Analytical Estimation of HI')

plt.subplot(424)
plt.imshow(HI_Pred_Full)
plt.colorbar()
#colorbar.set_label(r"${\rm log}(1+\delta_{\rm DM})$",fontsize=12,labelpad=0)
plt.title('Predicted HI')

plt.subplots_adjust(bottom=12, right=1.5, top=15)
plt.show()

f.savefig("Full_1.pdf")
