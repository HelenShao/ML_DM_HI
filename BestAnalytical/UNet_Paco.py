import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import sys, os
import numpy as np
import matplotlib.pyplot as plt


# This function implements the transformation we do to the data
def transform(x):
    return np.log(x+1.0) #x**0.25

################################ INPUT #######################################
root = '/mnt/ceph/users/fvillaescusa/Helen/maps'

maps = 10000
seed = 6    #random seed to split maps in train, valid and test sets

# Define NN hyperparameters
num_epochs    = 1000
batch_size    = 16
learning_rate = 0.001

f_best_model = 'BestModelDM_HI_log.pt'
##############################################################################

# Load estimated HI (input) maps 
HI_esti = np.load('HI_estimation.npy').astype('float32')
HI_esti = HI_esti.reshape(maps, 32, 32)

# Load HI and DM (True) maps and take mean & std
DM_True = np.zeros((maps, 32, 32), dtype=np.float32)
HI_True = np.zeros((maps, 32, 32), dtype=np.float32)
for i in range(maps):
    DM_True[i] = np.load('%s/DM_new_map_%d.npy'%(root,i))
    HI_True[i] = np.load('%s/HI_new_map_%d.npy'%(root,i))
      
# Take the log
DM_True = transform(DM_True)
HI_True = transform(HI_True)
HI_esti = transform(HI_esti)
  
# Mean and std values
DM_True_mean, DM_True_std = np.mean(DM_True), np.std(DM_True)
HI_True_mean, HI_True_std = np.mean(HI_True), np.std(HI_True)
HI_esti_mean, HI_esti_std = np.mean(HI_esti), np.std(HI_esti)

# Normalize Matrices
DM_True = (DM_True - DM_True_mean)/DM_True_std
HI_True = (HI_True - HI_True_mean)/HI_True_std
HI_esti = (HI_esti - HI_esti_mean)/HI_esti_std

# Convert to torch tensor
DM_True = torch.tensor(DM_True, dtype=torch.float)
HI_True = torch.tensor(HI_True, dtype=torch.float)
HI_esti = torch.tensor(HI_esti, dtype=torch.float)


######################## Create Datasets ######################
# Define Dataset 
class make_Dataset(Dataset):
    
    def __init__(self, name, seed):
         
        # shuffle the maps (instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342)
        np.random.seed(seed)
        indexes = np.arange(maps)
        np.random.shuffle(indexes)

        if   name=='train':  size, offset = 8000, 0
        elif name=='valid':  size, offset = 1000, 8000
        elif name=='test':   size, offset = 1000, 9000
        else:                raise Exception('Wrong name!')
        
        self.size      = size
        self.DM_True = np.zeros((size, 32, 32), dtype=np.float32)
        self.HI_True = np.zeros((size, 32, 32), dtype=np.float32)
        self.HI_esti = np.zeros((size, 32, 32), dtype=np.float32)
        
        # Load HI_estimation maps
        HI_esti = np.load('HI_estimation.npy')
        
        for i in range(size):
            
            # find the index of the map
            j = indexes[i+offset]

            # load maps
            self.DM_True[i] = np.load('%s/DM_new_map_%d.npy'%(root,j))
            self.HI_True[i] = np.load('%s/HI_new_map_%d.npy'%(root,j))
            self.HI_esti[i] = HI_esti[j].reshape(32,32)
            
            # Transform data
            self.DM_True[i] = transform(self.DM_True[i])
            self.HI_True[i] = transform(self.HI_True[i])
            self.HI_esti[i] = transform(self.HI_esti[i])
            
            # Normalize: (x-mean)/(std)
            self.DM_True[i] = (self.DM_True[i] - DM_True_mean) / DM_True_std
            self.HI_True[i] = (self.HI_True[i] - HI_True_mean) / HI_True_std
            self.HI_esti[i] = (self.HI_esti[i] - HI_esti_mean) / HI_esti_std
        
        # Convert into torch tensor
        self.DM_True = torch.tensor(self.DM_True, dtype=torch.float)
        self.HI_True = torch.tensor(self.HI_True, dtype=torch.float)
        self.HI_esti = torch.tensor(self.HI_esti, dtype=torch.float)
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.DM_True[idx], self.HI_True[idx], self.HI_esti[idx]

train_Dataset = make_Dataset('train', seed)
train_loader  = DataLoader(dataset=train_Dataset, batch_size=batch_size, 
                           shuffle=True)

valid_Dataset = make_Dataset('valid', seed)
valid_loader  = DataLoader(dataset=valid_Dataset, batch_size=batch_size, 
                           shuffle=True)

test_Dataset  = make_Dataset('test',  seed)
test_loader   = DataLoader(dataset=test_Dataset,  batch_size=batch_size, 
                           shuffle=True)


############################# Create UNet Model ################################
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self):
        super(UNet,self).__init__()
                
        self.dconv_down1 = double_conv(2, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', 
                                    align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(66, 1, 1)    
        #Change to 65 input channels because we add one more channel which is the orignal input image
        
        
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
        x12 = torch.cat([x11, conv1], dim=1)   
        
        x13 = self.dconv_up1(x12)
        x14 = torch.cat([x13, x], dim=1)   #Concatenate original image with last output
        out = self.conv_last(x14) 
        
        return out
        


################################# TRAINING LOOP ################################
# Call the model for training
model = UNet()

# Create loss and optimizer function for training  
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# load best-model
best_model = 9e7
if os.path.exists('BestModel_UNET_A.pt'):
    model.load_state_dict(torch.load('BestModel_UNet_A.pt'))

# main loop
for epoch in range(num_epochs):

    # TRAIN
    model.train()
    count, loss_train = 0, 0.0
    for DM_True, HI_True, HI_esti in train_loader:
        DM_True.unsqueeze_(1);  HI_True.unsqueeze_(1);  HI_esti.unsqueeze_(1)
        HI_input = torch.cat([DM_True, HI_esti], dim=1)
        
        # Forward Pass
        HI_pred = model(HI_input)
        loss    = criterion(HI_pred, HI_True)   
        loss_train += loss.detach().numpy()
        
        # Backward Prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        count += 1
    loss_train /= count
    
    # VALID
    model.eval() 
    count2, loss_valid = 0, 0.0
    for DM_True, HI_True, HI_esti in valid_loader:
        DM_True.unsqueeze_(1);  HI_True.unsqueeze_(1);  HI_esti.unsqueeze_(1)
        HI_input = torch.cat([DM_True, HI_esti], dim=1)

        HI_valid = model(HI_input)
        error    = criterion(HI_valid, HI_True)   
        loss_valid += error.detach().numpy()
        count2 += 1
    loss_valid /= count2
     
    # TEST
    model.eval() 
    count3, loss_test = 0, 0.0
    for DM_True, HI_True, HI_esti in test_loader:
        DM_True.unsqueeze_(1);  HI_True.unsqueeze_(1);  HI_esti.unsqueeze_(1)
        HI_input = torch.cat([DM_True, HI_esti], dim=1)
        
        HI_test  = model(HI_input)
        error    = criterion(HI_test, HI_True) 
        loss_test += error.detach().numpy()
        count3 += 1
    loss_test /= count3
    
    # Save Best Model 
    if loss_valid<best_model:
        best_model = loss_valid
        torch.save(model.state_dict(), 'BestModel_UNet_A.pt')

    # Print loss for both training and validation sets
    print('%03d %.4e %.4e %.4e'%(epoch, loss_train, loss_valid, loss_test))
    

################### Loss Function #######################
# Plot loss as a function of epochs
epochs = np.arange(num_epochs)
plt.plot(epochs, Loss_total, label = 'Loss_train')
plt.plot(epochs, loss_valid, label= 'Loss_Valid')
plt.plot(epochs, loss_test, label = 'Loss_Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

####################### Plot Images #########################
if os.path.exists('BestModel_UNET_A.pt'): 
    model.load_state_dict(torch.load('BestModel_UNET_A.pt'))
    
model.eval()

np.random.seed(seed)
indexes = np.arange(maps)
np.random.shuffle(indexes)

#Maps for 901
Map_in  = HI[indexes[901]]
Map_out = HI_True[indexes[901]]
plt.subplot(821)
hydrogen_pred1 = model(Map_in.unsqueeze_(0).unsqueeze_(0)).detach().numpy() 
maximum = np.max([np.max(hydrogen_pred1), torch.max(Map_out)])
minimum = np.min([np.min(hydrogen_pred1), torch.min(Map_out)])
plt.imshow(hydrogen_pred1[0,0,:,:], vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('Prediction Map of HI (901)')
loss = np.mean((hydrogen_pred1-HI_True[1].numpy())**2)
print('loss1 = %.3f'%loss)

plt.subplot(822)
plt.imshow(Map_out, vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('True Map of HI (901)')

plt.subplots_adjust(bottom=10, right=1.5, top=17)


#Maps for 902
Map_in  = HI[indexes[902]]
Map_out = HI_True[indexes[902]]
plt.subplot(823)
hydrogen_pred1 = model(Map_in.unsqueeze_(0).unsqueeze_(0)).detach().numpy() 
maximum = np.max([np.max(hydrogen_pred1), torch.max(Map_out)])
minimum = np.min([np.min(hydrogen_pred1), torch.min(Map_out)])
plt.imshow(hydrogen_pred1[0,0,:,:], vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('Prediction Map of HI (902)')
loss = np.mean((hydrogen_pred1-HI_True[1].numpy())**2)
print('loss2 = %.3f'%loss)

plt.subplot(824)
plt.imshow(Map_out, vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('True Map of HI (902)')

#Maps for 903
Map_in  = HI[indexes[903]]
Map_out = HI_True[indexes[903]]
plt.subplot(825)
hydrogen_pred1 = model(Map_in.unsqueeze_(0).unsqueeze_(0)).detach().numpy() 
maximum = np.max([np.max(hydrogen_pred1), torch.max(Map_out)])
minimum = np.min([np.min(hydrogen_pred1), torch.min(Map_out)])
print(np.max(hydrogen_pred1), np.min(hydrogen_pred1))
plt.imshow(hydrogen_pred1[0,0,:,:], vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('Prediction Map of HI (903)')
loss = np.mean((hydrogen_pred1-HI_True[1].numpy())**2)
print('loss3 = %.3f'%loss)

plt.subplot(826)
plt.imshow(Map_out, vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('True Map of HI (903)')

#Maps for 4
Map_in  = HI[indexes[904]]
Map_out = HI_True[indexes[904]]
plt.subplot(827)
hydrogen_pred1 = model(Map_in.unsqueeze_(0).unsqueeze_(0)).detach().numpy() 
maximum = np.max([np.max(hydrogen_pred1), torch.max(Map_out)])
minimum = np.min([np.min(hydrogen_pred1), torch.min(Map_out)])
plt.imshow(hydrogen_pred1[0,0,:,:], vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('Prediction Map of HI (902)')
loss = np.mean((hydrogen_pred1-HI_True[1].numpy())**2)
print('loss4 = %.3f'%loss)

plt.subplot(828)
plt.imshow(Map_out, vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('True Map of HI (904)')

