import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import sys, os

import numpy as np
import matplotlib.pyplot as plt


# Define hyperparameters

num_epochs = 500
batch_size = 16
learning_rate = 0.001

# Define Train Dataset 

class Train_Dataset(Dataset):
    
    def __init__(self):
        self.DM_matrix_train = np.zeros((800, 4096), dtype = np.float32)
        self.HI_matrix_train = np.zeros((800, 1024), dtype = np.float32)
        
        for i in range(800):
            self.DM_matrix_train[i] = np.load('DM_map_%d.npy'%(i)).flatten()
            self.HI_matrix_train[i] = np.load('HI_map_%d.npy'%(i)).flatten()
            
            # Take the log
            self.DM_matrix_train[i] = np.log(self.DM_matrix_train[i] + 1)
            self.HI_matrix_train[i] = np.log(self.HI_matrix_train[i] + 1)
        
        self.DM_matrix_train = torch.tensor(self.DM_matrix_train, dtype=torch.float)
        self.HI_matrix_train = torch.tensor(self.HI_matrix_train, dtype=torch.float)

        
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
        self.DM_matrix_valid = np.zeros((100, 4096), dtype = np.float32)
        self.HI_matrix_valid = np.zeros((100, 1024), dtype = np.float32)
        
        for i in range(100):
            self.DM_matrix_valid[i] = np.load('DM_map_%d.npy'%(800+i)).flatten()
            self.HI_matrix_valid[i] = np.load('HI_map_%d.npy'%(800+i)).flatten()
            
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

DM_matrix_test = np.zeros((100, 4096), dtype = np.float32)
HI_matrix_test = np.zeros((100, 1024), dtype = np.float32)

for i in range(100):
    DM_matrix_test[i] = np.load('DM_map_%d.npy'%(900+i)).flatten()
    HI_matrix_test[i] = np.load('HI_map_%d.npy'%(900+i)).flatten()
            
    # Take the log
    DM_matrix_test[i] = np.log(DM_matrix_test[i] + 1)
    HI_matrix_test[i] = np.log(HI_matrix_test[i] + 1)

DM_matrix_test = torch.tensor(DM_matrix_test, dtype=torch.float)
HI_matrix_test = torch.tensor(HI_matrix_test, dtype=torch.float)
    
print(DM_matrix_test.shape)
print(HI_matrix_test.shape)

=== Model: Fully Connected - NN ===

# Create Model Container: 3 fully connected layers (linear)
Model = nn.Sequential()

# Define Layers
fc1 = nn.Linear(4096,100)      #Input size is the size of channels for the DM maps
Relu1 = nn.ReLU()
fc2 = nn.Linear(100,200)      
Relu2 = nn.ReLU()
fc3 = nn.Linear(200,1024)      #Output size is the size of channels for HI maps
Relu3 = nn.ReLU()

#Add layers to Model

Model.add_module('Lin1', fc1)
Model.add_module('Relu1', Relu1)
Model.add_module('fc2', fc2)
Model.add_module('Relu2', Relu2)
Model.add_module('fc3', fc3)
Model.add_module('Relu3', Relu3)

=== Train Model ===

criterion = nn.MSELoss()       #Loss Function           
optimizer = torch.optim.Adam(Model.parameters(), lr = learning_rate)

##### Training Loop #####

Loss_total = np.zeros(500) # Record loss for plotting
loss_valid = np.zeros(500)  #Record Validation loss for saving
best_model = 9e7

# load best-model
#if os.path.exists('BestModelDM_HI.pt'):
#    Model.load_state_dict(torch.load('BestModelDM_HI.pt'))

for epoch in range(num_epochs):
    partial_loss = 0.0
    for i, (DM_matrix_train, HI_matrix_train) in enumerate(train_loader):
      
        #Forward Pass
        Model.train()
        HI_pred = Model(DM_matrix_train)
        
        loss = criterion(HI_pred, HI_matrix_train)   #Loss  for train set
        Loss_total_saved[epoch] = loss.detach().numpy()
        partial_loss += loss.detach().numpy()
        
        #Backward Prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    for i, (DM_matrix_valid, HI_matrix_valid) in enumerate(validation_loader):
        Model.eval()  #Set model into eval mode to stop back prop
        HI_validation = Model(DM_matrix_valid)
        error_valid = criterion(HI_validation, HI_matrix_valid)   #Loss for validation set
        loss_valid[epoch] = error_valid.detach().numpy()
    
    #Save Best Model 
    if loss_valid[epoch]<best_model:
        best_model = loss_valid[epoch]
        torch.save(Model.state_dict(), 'BestModelDM_HI.pt')

    #Print loss for both training and validation sets
    Loss_total[epoch] = partial_loss    
    print('Epoch:', epoch, 'Loss: ', Loss_total[epoch], '    Valid_Error:', loss_valid[epoch])
    
==== Evaluate ====
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

==== Test Model ====
# Plot images of neutral hydrogen prediction vs true image
plt.subplot(821)
plt.imshow(HI_matrix_test[1].reshape((32,32)))
plt.title('True Map of HI (801)')

plt.subplot(822)
hydrogen_pred1 = Model(DM_matrix_test[1]) 
plt.imshow(hydrogen_pred1.detach().numpy().reshape((32,32)))
plt.title('Prediction Map of HI (801)')

plt.subplot(823)
plt.imshow(HI_matrix_test[2].reshape((32,32)))
plt.title('True Map of HI (802)')

plt.subplot(824)
hydrogen_pred2 = Model(DM_matrix_test[2]) 
plt.imshow(hydrogen_pred2.detach().numpy().reshape((32,32)))
plt.title('Prediction Map of HI (802)')

plt.subplot(825)
plt.imshow(HI_matrix_test[3].reshape((32,32)))
plt.title('True Map of HI (803)')

plt.subplot(826)
hydrogen_pred3 = Model(DM_matrix_test[3]) 
plt.imshow(hydrogen_pred3.detach().numpy().reshape((32,32)))
plt.title('Prediction Map of HI (803)')

plt.subplot(827)
plt.imshow(HI_matrix_test[4].reshape((32,32)))
plt.title('True Map of HI (804)')

plt.subplot(828)
hydrogen_pred4 = Model(DM_matrix_test[4]) 
plt.imshow(hydrogen_pred4.detach().numpy().reshape((32,32)))
plt.title('Prediction Map of HI (804)')

plt.subplots_adjust(bottom=10, right=1.5, top=17)
plt.show()
