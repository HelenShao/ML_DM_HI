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

# This function implements the transformation we do to the data
def transform(x):
    return np.log(x+1.0) #x**0.25

################################ INPUT #######################################
maps = 1000
seed = 6    #random seed to split maps in train, valid and test sets

# Define NN hyperparameters
num_epochs    = 100
batch_size    = 16
learning_rate = 0.0005

##############################################################################

# Load HI (True) maps and take mean & std
HI_True = np.zeros((maps, 32, 32), dtype=np.float32)

#Load HI (input) maps 
HI = np.load('HI_estimation.npy').astype('float32')
HI = HI.reshape(1000, 32, 32)

# do a loop over all maps
for i in range(maps):
    HI_True[i] = np.load('HI_map_%d.npy'%(i))
      
    # Take the log
    HI_True[i] = transform(HI_True[i])
    HI[i] = transform(HI[i])
    #HI_True[i], HI[i] = np.log(HI_True[i] + 1.0), np.log(HI[i] + 1.0)
    

# Mean and std values
HI_True_mean, HI_True_std = np.mean(HI_True), np.std(HI_True)
HI_mean, HI_std = np.mean(HI), np.std(HI)


# Print mean and std values
print(HI_True_mean, HI_True_std)
print(HI_mean, HI_std)

#Normalize Matricies
HI_True = (HI_True - HI_True_mean)/HI_True_std
HI = (HI - HI_mean)/HI_std

#Convert to torch tensor
HI_True = torch.tensor(HI_True, dtype=torch.float)
HI = torch.tensor(HI, dtype=torch.float)


############# Create Datasets ############
# Define Dataset 
class make_Dataset(Dataset):
    
    def __init__(self, name, seed):
         
        # shuffle the maps (instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342)
        np.random.seed(seed)
        indexes = np.arange(maps)
        np.random.shuffle(indexes)

        if   name=='train':  size, offset = 800, 0
        elif name=='valid':  size, offset = 100, 800
        elif name=='test':   size, offset = 100, 900
        else:                raise Exception('Wrong name!')
        
        self.size      = size
        self.HI_True = np.zeros((size, 32, 32), dtype=np.float32)
        self.HI = np.zeros((size, 32, 32), dtype=np.float32)
        
        #Load HI_estimation maps
        HI_estimation = np.load('HI_estimation.npy')
        
        for i in range(size):
            
            # find the index of the map
            j = indexes[i+offset]

            # load maps
            self.HI_True[i] = np.load('HI_map_%d.npy'%(j))
            self.HI[i] = HI_estimation[j].reshape(32,32)
            
            # Transform data
            self.HI_True[i] = transform(self.HI_True[i])
            self.HI[i] = transform(self.HI[i])
            #self.Hi_True[i] = np.log(self.HI_True[i] + 1.0)
            #self.HI[i] = np.log(self.HI[i] + 1.0)
            
            # Normalize: (x-mean)/(std)
            self.HI_True[i] = (self.HI_True[i] - HI_True_mean) / HI_True_std
            self.HI[i] = (self.HI[i] - HI_mean) / HI_std
        
        # Convert into torch tensor
        self.HI_True = torch.tensor(self.HI_True, dtype=torch.float)
        self.HI = torch.tensor(self.HI, dtype=torch.float)
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.HI_True[idx], self.HI[idx]

train_Dataset = make_Dataset('train', seed)
train_loader  = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle=True)

valid_Dataset = make_Dataset('valid', seed)
valid_loader  = DataLoader(dataset=valid_Dataset, batch_size=batch_size, shuffle=True)

test_Dataset  = make_Dataset('test',  seed)
test_loader   = DataLoader(dataset=test_Dataset,  batch_size=batch_size, shuffle=True)

class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        #Branch1x1 is a single 1x1 Conv
        
        self.branch5x5_1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        #Branch5x5  has 2 conv layers, first of 1x1 and second of 5x5
        
        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        #Branch3x3 has 1x1, 3x3, and 3x3.
        
        self.branch_pool = nn.Conv2d(in_channels, 32, kernel_size=1)
        #Branch_pool has avg_pool and 1x1
        
        #Dimensions are kept same for all branches! 
        
    def forward(self, x):      #Pass input through the branches separately
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)
        #Concatenate the outputs

class Model(nn.Module): #Define main model

    def __init__(self):
        super(Model, self).__init__()
        self.Conv5x5 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, stride = 1, padding = 2)
        self.incept1 = InceptionB(in_channels= 1)
        self.Conv5x5_2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 5, stride = 1, padding = 2)
        self.incept2 = InceptionB(in_channels = 256)
        self.Conv5x5_3 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 5, stride = 1, padding = 2)
        self.incept3 = InceptionB(in_channels = 256)
        self.Conv1x1 = nn.Conv2d(in_channels = 256, out_channels = 1, kernel_size = 1)
        """The in_channels for Conv1x1 is the sum of the out_channels from
        incept1 because you concatenated the out_channels in incept1"""
        
        #self.Dropout = nn.Dropout()
        #self.fc1 = nn.Linear(16384, 1000)
        #self.fc2 = nn.Linear(1000, 1024)

    def forward(self, x):
        in_size = x.size(0)
        #x = self.Conv5x5(x)
        #x = F.relu(x)
        x = self.incept1(x)
        x = F.relu(x)
        #x = self.Conv5x5_2(x)
        #x = F.relu(x)
        #x = self.incept2(x)
        #x = F.relu(x)
        #x = self.Conv5x5_3(x)
        #x = F.relu(x)
        #x = self.incept3(x)
        #x = F.relu(x)
        x = self.Conv1x1(x)
        return(x)


# Call the model for training

model = Model()

# Create loss and optimizer function for training  
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)  #Updates Parameters

#Train Model
Loss_total = np.zeros(100) # Record loss for plotting
loss_valid = np.zeros(100)  #Record Validation loss for saving
loss_test = np.zeros(100)
best_model = 9e7

# load best-model
#if os.path.exists('BestModel_InceptionB_A.pt'):
    #model.load_state_dict(torch.load('BestModel_InceptionB_A.pt'))

for epoch in range(num_epochs):
    partial_loss = 0.0
    partial_loss_valid = 0.0
    partial_loss_test = 0.0
    count = 0
    for i, (HI, HI_True) in enumerate(train_loader):
        HI = HI.unsqueeze(1)  #Add extra dimension
        HI_True.unsqueeze_(1)    #Remove extra dimension
        
        #Forward Pass
        model.train()
        HI_pred = model(HI)
        
        loss = criterion(HI_pred, HI_True)   
        partial_loss += loss.detach().numpy()
        
        #Backward Prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        count += 1
    partial_loss = partial_loss / count
    
    count2 = 0
    for i, (HI, HI_True) in enumerate(valid_loader):
        model.eval()  #Set model into eval mode to stop back prop
        
        HI = HI.unsqueeze(1)
        HI_True.unsqueeze_(1)

        HI_validation = model(HI)
        error_valid = criterion(HI_validation, HI_True)   #Loss for validation set
        partial_loss_valid += error_valid.detach().numpy()
        count2 += 1
    partial_loss_valid = partial_loss_valid / count2
     
    count3 = 0
    for i, (HI, HI_True) in enumerate(test_loader):
        model.eval()  #Set model into eval mode to stop back prop
        
        HI = HI.unsqueeze(1)
        HI_True.unsqueeze_(1)
        
        HI_test = model(HI)
        error_test = criterion(HI_test, HI_True)   #Loss for validation set
        partial_loss_test += error_valid.detach().numpy()
        count3 += 1
    partial_loss_test = partial_loss_test / count2
    
    #Save Best Model 
    if partial_loss_valid<best_model:
        best_model = loss_valid[epoch]
        torch.save(model.state_dict(), 'BestModel_InceptionB_1A.pt')

    #Print loss for both training and validation sets
    Loss_total[epoch] = partial_loss
    loss_valid[epoch] = partial_loss_valid
    loss_test[epoch] = partial_loss_test
    print('Epoch:', epoch,  'Train: ', Loss_total[epoch], ' Valid:', loss_valid[epoch], 'Test:', loss_test[epoch])
    
    
################### Loss Function #######################
# Plot loss as a function of epochs
epochs = np.arange(100)
plt.plot(epochs, Loss_total, label = 'Loss_train')
plt.plot(epochs, loss_valid, label= 'Loss_Valid')
plt.plot(epochs, loss_test, label = 'Loss_Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

####################### Plot Images ############################

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
