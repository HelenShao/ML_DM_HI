import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import sys, os
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt


# Define hyperparameters

num_epochs = 500
batch_size = 16
learning_rate = 0.00001


# Define Train Dataset 

class Train_Dataset(Dataset):
    
    def __init__(self):
        self.DM_matrix_train = np.zeros((800, 32, 32), dtype = np.float32)
        self.HI_matrix_train = np.zeros((800, 32, 32), dtype = np.float32)
        
        for i in range(800):
            self.DM_matrix_train[i] = np.load('DM_new_map_%d.npy'%(i))
            self.HI_matrix_train[i] = np.load('HI_new_map_%d.npy'%(i))
            
            # Take the log
            self.DM_matrix_train[i] = np.log(self.DM_matrix_train[i] + 1)
            self.HI_matrix_train[i] = np.log(self.HI_matrix_train[i] + 1)
        
        self.DM_matrix_train = torch.tensor(self.DM_matrix_train, dtype=torch.float)
        self.HI_matrix_train = torch.tensor(self.HI_matrix_train, dtype=torch.float)

        self.DM_matrix_train.unsqueeze(1)
      
        """ For i in range 1000 (there are 800 maps for each) map_i will 
            be loaded into row i of the matrix. 800 rows, each row has either 64x64 or 32x32 size"""
            
    def __len__(self):
            return len(self.DM_matrix_train)

    def __getitem__(self, idx):
        return self.DM_matrix_train[idx], self.HI_matrix_train[idx]
    
train_Dataset = Train_Dataset()
train_loader = DataLoader(dataset= train_Dataset, batch_size = batch_size, shuffle = True)


# Define Validation Dataset 

class Validation_Dataset(Dataset):
    
    def __init__(self):
        self.DM_matrix_valid = np.zeros((100, 32, 32), dtype = np.float32)
        self.HI_matrix_valid = np.zeros((100, 32, 32), dtype = np.float32)
        
        for i in range(100):
            self.DM_matrix_valid[i] = np.load('DM_new_map_%d.npy'%(800+i))
            self.HI_matrix_valid[i] = np.load('HI_new_map_%d.npy'%(800+i))
            
            # Take the log
            self.DM_matrix_valid[i] = np.log(self.DM_matrix_valid[i] + 1)
            self.HI_matrix_valid[i] = np.log(self.HI_matrix_valid[i] + 1)
        
        self.DM_matrix_valid = torch.tensor(self.DM_matrix_valid, dtype=torch.float)
        self.HI_matrix_valid = torch.tensor(self.HI_matrix_valid, dtype=torch.float)

        
        """ For i in range 1000 (there are 800 maps for each) map_i will 
            be loaded into row i of the matrix. 800 rows, each row has either 64x64 or 32x32 size"""
            
    def __len__(self):
            return len(self.DM_matrix_valid)

    def __getitem__(self, idx):
        return self.DM_matrix_valid[idx], self.HI_matrix_valid[idx]

valid_Dataset = Validation_Dataset()
validation_loader = DataLoader(dataset= valid_Dataset, batch_size = batch_size, shuffle = True)


# Define Test Dataset 

DM_matrix_test = np.zeros((100, 1, 32, 32), dtype = np.float32)
HI_matrix_test = np.zeros((100, 1, 32, 32), dtype = np.float32)

for i in range(100):
    DM_matrix_test[i,0] = np.load('DM_new_map_%d.npy'%(900+i))
    HI_matrix_test[i,0] = np.load('HI_new_map_%d.npy'%(900+i))
    
    # Take the log
    DM_matrix_test[i,0] = np.log(DM_matrix_test[i] + 1)
    HI_matrix_test[i,0] = np.log(HI_matrix_test[i] + 1)

DM_matrix_test = torch.tensor(DM_matrix_test, dtype=torch.float)
HI_matrix_test = torch.tensor(HI_matrix_test, dtype=torch.float)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2. This is used in the downsampling blocks'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return(x)


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return(x)
        
        
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)   #Copy and crop - concatenate 
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return(F.relu(x))
 
# Call the model for training

model = UNet(1, 1)

# Create loss and optimizer function for training  
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)  #Updates Parameters



#Train Model

Loss_total = np.zeros(500) # Record loss for plotting
loss_valid = np.zeros(500)  #Record Validation loss for saving
best_model = 9e7

# load best-model
if os.path.exists('BestModelDM_HI_CNN_UNet.pt'):
    model.load_state_dict(torch.load('BestModelDM_HI_UNet.pt'))

for epoch in range(num_epochs):
    partial_loss = 0.0
    partial_loss_valid = 0.0
    count = 0
    for i, (DM_matrix_train, HI_matrix_train) in enumerate(train_loader):
        DM_matrix_train= DM_matrix_train.unsqueeze(1)
        HI_matrix_train.squeeze_()
        
        #Forward Pass
        model.train()
        HI_pred = model(DM_matrix_train)
        
        #HI_pred = Variable(torch.randn(10, 120).float(), requires_grad = True)
        #HI_matrix_train = Variable(torch.FloatTensor(10).uniform_(0, 120).long())
        
        loss = criterion(HI_pred, HI_matrix_train)   #Loss  for train set
        #Loss_total[epoch] = loss.detach().numpy()
        partial_loss += loss.detach().numpy()
        
        #Backward Prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #after 250 epochs load the best model and decrease the learning rate
        if epoch==250:
            #model.load_state_dict(torch.load('BestModelDM_HI_UNet.pt'))
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/2)
        count += 1
    partial_loss = partial_loss / count

    count2 = 0
    for i, (DM_matrix_valid, HI_matrix_valid) in enumerate(validation_loader):
        model.eval()  #Set model into eval mode to stop back prop
        
        DM_matrix_valid= DM_matrix_valid.unsqueeze(1)
        HI_matrix_valid.squeeze_()
        
        HI_validation = model(DM_matrix_valid)
        
        #HI_validation = Variable(torch.randn(10, 120).float(), requires_grad = True)
        #HI_matrix_valid = Variable(torch.FloatTensor(10).uniform_(0, 120).long())
        
        error_valid = criterion(HI_validation, HI_matrix_valid)   #Loss for validation set
        partial_loss_valid += error_valid
        count2 += 1
    partial_loss_valid = partial_loss_valid / count2
     
    #Save Best Model 
    if loss_valid[epoch]<best_model:
        best_model = loss_valid[epoch]
        torch.save(model.state_dict(), 'BestModelDM_HI_UNet.pt')

    #Print loss for both training and validation sets
    Loss_total[epoch] = partial_loss
    loss_valid[epoch] = partial_loss_valid
    print('Epoch:', epoch,  'Loss: ', Loss_total[epoch], '    Valid_Error:', loss_valid[epoch])
    
# Plot loss as a function of epochs
epochs = np.arange(500)

plt.plot(epochs, Loss_total, label = 'Loss_train')
plt.plot(epochs, loss_valid, label= 'Loss_Valid')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
