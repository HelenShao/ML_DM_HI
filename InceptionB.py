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
learning_rate = 0.001


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
        
        print(self.DM_matrix_train.shape)
        """ For i in range 1000 (there are 800 maps for each) map_i will 
            be loaded into row i of the matrix. 800 rows, each row has 32x32 map"""
            
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

DM_matrix_test = np.zeros((100, 32, 32), dtype = np.float32)
HI_matrix_test = np.zeros((100, 32, 32), dtype = np.float32)

for i in range(100):
    DM_matrix_test[i] = np.load('DM_new_map_%d.npy'%(900+i))
    HI_matrix_test[i] = np.load('HI_new_map_%d.npy'%(900+i))
            
    # Take the log
    DM_matrix_test[i] = np.log(DM_matrix_test[i] + 1)
    HI_matrix_test[i] = np.log(HI_matrix_test[i] + 1)

DM_matrix_test = torch.tensor(DM_matrix_test, dtype=torch.float)
HI_matrix_test = torch.tensor(HI_matrix_test, dtype=torch.float)
    
print(DM_matrix_test.shape)
print(HI_matrix_test.shape)


###### Creating the Model ######

class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        #Branch1x1 is a single 1x1 Conv
        
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(in_channels, 24, kernel_size=5, padding=2)
        #Branch5x5  has 2 conv layers, first of 1x1 and second of 5x5
        
        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        #Branch3x3 has 1x1, 3x3, and 3x3.
        
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
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
        self.Conv1x1_1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 1)
        self.incept1 = InceptionB(in_channels= 16)
        self.Conv1x1_2 = nn.Conv2d(in_channels = 88, out_channels = 1, kernel_size = 1)
        """The in_channels for Conv1x1 is the sum of the out_channels from
        incept1 because you concatenated the out_channels in incept1"""

        #self.fc1 = nn.Linear(16384, 1000)
        #self.fc2 = nn.Linear(1000, 1024)

    def forward(self, x):
        in_size = x.size(0)
        x = self.Conv1x1_1(x)
        x = self.incept1(x)
        x = F.relu(x)
        x = self.Conv1x1_2(x)
        x = F.relu(x)
        #x = x.view(in_size, -1)  # flatten the tensor
        #x = self.fc1(x)
        #x = self.fc2(x)
        return(x)


# Call the model for training
model = Model()

# Create loss and optimizer function for training  
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)  #Updates Parameters


####### Train Model ######

Loss_total = np.zeros(500) # Record loss for plotting
loss_valid = np.zeros(500)  #Record Validation loss for saving
best_model = 9e7

# load best-model
if os.path.exists('BestModelDM_HI_CNN_InpcetionB.pt'):
    model.load_state_dict(torch.load('BestModelDM_HI_InpcetionB.pt'))

for epoch in range(num_epochs):
    partial_loss = 0.0
    partial_loss_valid = 0.0
    for i, (DM_matrix_train, HI_matrix_train) in enumerate(train_loader):
        DM_matrix_train= DM_matrix_train.unsqueeze(1)
        HI_matrix_train.squeeze_()
        
        #Forward Pass
        model.train()
        HI_pred = model(DM_matrix_train)
        
        #HI_pred = Variable(torch.randn(10, 120).float(), requires_grad = True)
        #HI_matrix_train = Variable(torch.FloatTensor(10).uniform_(0, 120).long())
        
        loss = criterion(HI_pred, HI_matrix_train)   #Loss  for train set
        Loss_total[epoch] = loss.detach().numpy()
        partial_loss += loss.detach().numpy()
        partial_loss = partial_loss / batch_size
        
        #Backward Prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # after 250 epochs load the best model and decrease the learning rate
        if epoch==250:
            model.load_state_dict(torch.load('BestModelDM_HI_InceptionB.pt'))
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/2)
            
    for i, (DM_matrix_valid, HI_matrix_valid) in enumerate(validation_loader):
        model.eval()  #Set model into eval mode to stop back prop
        DM_matrix_valid= DM_matrix_valid.unsqueeze(1)
        HI_matrix_valid.squeeze_()
        HI_validation = model(DM_matrix_valid)
        
        #HI_validation = Variable(torch.randn(10, 120).float(), requires_grad = True)
        #HI_matrix_valid = Variable(torch.FloatTensor(10).uniform_(0, 120).long())
        
        error_valid = criterion(HI_validation, HI_matrix_valid)   #Loss for validation set
        partial_loss_valid += error_valid
        partial_loss_valid = partial_loss_valid / batch_size
        
    #Save Best Model 
    if loss_valid[epoch]<best_model:
        best_model = loss_valid[epoch]
        torch.save(model.state_dict(), 'BestModelDM_HI_InceptionB.pt')

    #Print loss for both training and validation sets
    Loss_total[epoch] = partial_loss  
    loss_valid[epoch] = partial_loss_valid
    print('Epoch:', epoch, 'Loss: ', Loss_total[epoch], '    Valid_Error:', loss_valid[epoch])
    
    
# Plot loss as a function of epochs
epochs = np.arange(500)

plt.plot(epochs, Loss_total, label = 'Loss_train')
plt.plot(epochs, loss_valid, label= 'Loss_Valid')
plt.plot(epochs, loss_test, label = 'Loss_Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
