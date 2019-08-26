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

num_epochs = 3000
batch_size = 16
learning_rate = 0.0005

#Find mean and standard deviation for data sets

#Train Dataset
DM_train = np.zeros((800, 1024), dtype = np.float32)
HI_train = np.zeros((800, 1024), dtype = np.float32)

for i in range(800):
    DM_train[i] = np.load('DM_new_map_%d.npy'%(i)).flatten()
    HI_train[i] = np.load('HI_new_map_%d.npy'%(i)).flatten()
            
    # Take the log
    DM_train[i] = np.log(DM_train[i] + 1)
    HI_train[i] = np.log(HI_train[i] + 1)
    
#Mean Values
DM_train_mean = np.mean(DM_train)
HI_train_mean = np.mean(HI_train)

#STD Values
DM_train_std = np.std(DM_train)
HI_train_std = np.std(HI_train)


#Validation Dataset
DM_valid = np.zeros((100, 1024), dtype = np.float32)
HI_valid = np.zeros((100, 1024), dtype = np.float32)

for i in range(100):
    DM_valid[i] = np.load('DM_new_map_%d.npy'%(800+i)).flatten()
    HI_valid[i] = np.load('HI_new_map_%d.npy'%(800+i)).flatten()
            
    # Take the log
    DM_valid[i] = np.log(DM_valid[i] + 1)
    HI_valid[i] = np.log(HI_valid[i] + 1)
    
#Mean Values
DM_valid_mean = np.mean(DM_valid)
HI_valid_mean = np.mean(HI_valid)

#STD Values
DM_valid_std = np.std(DM_valid)
HI_valid_std = np.std(HI_valid)


#Test Dataset
DM_test = np.zeros((100, 1024), dtype = np.float32)
HI_test = np.zeros((100, 1024), dtype = np.float32)

for i in range(100):
    DM_test[i] = np.load('DM_new_map_%d.npy'%(900+i)).flatten()
    HI_test[i] = np.load('HI_new_map_%d.npy'%(900+i)).flatten()
            
    # Take the log
    DM_test[i] = np.log(DM_test[i] + 1)
    HI_test[i] = np.log(HI_test[i] + 1)
    
#Mean Values
DM_test_mean = np.mean(DM_test)
HI_test_mean = np.mean(HI_test)

#STD Values
DM_test_std = np.std(DM_test)
HI_test_std = np.std(HI_test)

###### Create Datasets ######

# Define Train Dataset 

class Train_Dataset(Dataset):
    
    def __init__(self):
        self.DM_matrix_train = np.zeros((800, 1024), dtype = np.float32)
        self.HI_matrix_train = np.zeros((800, 1024), dtype = np.float32)
        
        for i in range(800):
            self.DM_matrix_train[i] = np.load('DM_new_map_%d.npy'%(i)).flatten()
            self.HI_matrix_train[i] = np.load('HI_new_map_%d.npy'%(i)).flatten()
            
            # Take the log
            self.DM_matrix_train[i] = np.log(self.DM_matrix_train[i] + 1)
            self.HI_matrix_train[i] = np.log(self.HI_matrix_train[i] + 1)
            
            #Normalize: (x-mean)/(std)
            self.DM_matrix_train[i] = (self.DM_matrix_train[i] - DM_train_mean) / (DM_train_std)
            self.HI_matrix_train[i] = (self.HI_matrix_train[i] - HI_train_mean) / (HI_train_std)
        
        #Convert into torch tensor
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
        self.DM_matrix_valid = np.zeros((100, 1024), dtype = np.float32)
        self.HI_matrix_valid = np.zeros((100, 1024), dtype = np.float32)
        
        for i in range(100):
            self.DM_matrix_valid[i] = np.load('DM_new_map_%d.npy'%(800+i)).flatten()
            self.HI_matrix_valid[i] = np.load('HI_new_map_%d.npy'%(800+i)).flatten()
            
            # Take the log
            self.DM_matrix_valid[i] = np.log(self.DM_matrix_valid[i] + 1)
            self.HI_matrix_valid[i] = np.log(self.HI_matrix_valid[i] + 1)
            
            #Normalize: (x-mean)/(std)
            self.DM_matrix_valid[i] = (self.DM_matrix_valid[i] - DM_valid_mean) / (DM_valid_std)
            self.HI_matrix_valid[i] = (self.HI_matrix_valid[i] - HI_valid_mean) / (HI_valid_std)
        
        #Convert into torch tensor
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

DM_matrix_test = np.zeros((100, 1024), dtype = np.float32)
HI_matrix_test = np.zeros((100, 1024), dtype = np.float32)

for i in range(100):
    DM_matrix_test[i] = np.load('DM_new_map_%d.npy'%(900+i)).flatten()
    HI_matrix_test[i] = np.load('HI_new_map_%d.npy'%(900+i)).flatten()
            
    # Take the log
    DM_matrix_test[i] = np.log(DM_matrix_test[i] + 1)
    HI_matrix_test[i] = np.log(HI_matrix_test[i] + 1)
    
    #Normalize: (x-mean)/(std)
    DM_matrix_test[i] = (DM_matrix_test[i] - DM_test_mean) / (DM_test_std)
    HI_matrix_test[i] = (HI_matrix_test[i] - HI_test_mean) / (HI_test_std)
        
#Convert into torch tensor
DM_matrix_test = torch.tensor(DM_matrix_test, dtype=torch.float)
HI_matrix_test = torch.tensor(HI_matrix_test, dtype=torch.float)


# Create Model Container: 3 fully connected layers (linear)
Model = nn.Sequential()

# Define Layers
fc1 = nn.Linear(1024,500)      #Input size is the size of channels for the DM maps
Relu1 = nn.ReLU()
fc2 = nn.Linear(500,500)      
Relu2 = nn.ReLU()
fc3 = nn.Linear(500,1024)      #Output size is the size of channels for HI maps
Relu3 = nn.ReLU()

#Add layers to Model

Model.add_module('Lin1', fc1)
Model.add_module('Relu1', Relu1)
Model.add_module('fc2', fc2)
Model.add_module('Relu2', Relu2)
Model.add_module('fc3', fc3)
Model.add_module('Relu3', Relu3)

#Train Model

criterion = nn.MSELoss()       #Loss Function           
optimizer = torch.optim.Adam(Model.parameters(), lr = learning_rate)

#Train the Model

criterion = nn.MSELoss()       #Loss Function           
optimizer = torch.optim.Adam(Model.parameters(), lr = learning_rate)

##### Training Loop #####

Loss_total = np.zeros(3000) # Record loss for plotting
loss_valid = np.zeros(3000)  #Record Validation loss for saving
best_model = 9e7

# load best-model
if os.path.exists('BestModelDM_HI_New_2.pt'):
    Model.load_state_dict(torch.load('BestModelDM_HI_New_2.pt'))

for epoch in range(num_epochs):
    partial_loss = 0.0
    partial_loss_valid = 0.0
    count = 0
    
    #### Decrease Loss ####
    # after 250 epochs load the best model and decrease the learning rate
    if epoch==250:
        Model.load_state_dict(torch.load('BestModelDM_HI_New_2.pt'))
        optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate/2)
        
    # after 500 epochs load the best model and decrease the learning rate
    if epoch==500:
        Model.load_state_dict(torch.load('BestModelDM_HI_New_2.pt'))
        optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate/5)
            
    # After 1000 epochs, load the best model and decrease the learning rate
    if epoch==1000:
        Model.load_state_dict(torch.load('BestModelDM_HI_New_2.pt'))
        optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate/10)
            
        # After 1250 epochs, load the best model and decrease the learning rate
    if epoch==1250:
        Model.load_state_dict(torch.load('BestModelDM_HI_New_2.pt'))
        optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate/15)
            
    for i, (DM_matrix_train, HI_matrix_train) in enumerate(train_loader):
        #Forward Pass
        Model.train()
        HI_pred = Model(DM_matrix_train)
        
        loss = criterion(HI_pred, HI_matrix_train)   #Loss  for train set
        #Loss_total[epoch] = loss.detach().numpy()
        partial_loss += loss.detach().numpy()
        
        #Backward Prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        count += 1
    partial_loss = partial_loss/count  #Divide partial_loss by number of iterations
       
    #Validation Set
    count2 = 0
    for i, (DM_matrix_valid, HI_matrix_valid) in enumerate(validation_loader):
        Model.eval()  #Set model into eval mode to stop back prop
        HI_validation = Model(DM_matrix_valid)
        error_valid = criterion(HI_validation, HI_matrix_valid)   #Loss for validation set
        partial_loss_valid += error_valid
        count2 += 1
        
    partial_loss_valid = partial_loss_valid / count2
    
    #Save Best Model 
    if loss_valid[epoch]<best_model:
        best_model = loss_valid[epoch]
        torch.save(Model.state_dict(), 'BestModelDM_HI_New_2.pt')

    #Print loss for both training and validation sets
    Loss_total[epoch] = partial_loss  
    loss_valid[epoch] = partial_loss_valid
    print('Epoch:', epoch, 'Loss: ', Loss_total[epoch], ' Valid_Error:', loss_valid[epoch])
    

torch.save(Model.state_dict(), 'BestModelDM_HI_New_2.pt')


# Plot loss as a function of epochs
epochs = np.arange(3000)

plt.plot(epochs, Loss_total, label = 'Loss_train')
plt.plot(epochs, loss_valid, label= 'Loss_Valid')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

#################### Predictions #######################

# Plot images of neutral hydrogen prediction vs true image
if os.path.exists('BestModelDM_HI_New.pt'):
    Model.load_state_dict(torch.load('BestModelDM_HI_New.pt'))

Model.eval()

plt.subplot(821)
hydrogen_pred1 = Model(DM_matrix_test[1]).detach().numpy() 
maximum = np.max([np.max(hydrogen_pred1), torch.max(HI_matrix_test[1])])
plt.imshow(hydrogen_pred1.reshape((32,32)), vmin = 0, vmax = maximum)
plt.colorbar()
plt.title('Prediction Map of HI (901)')

plt.subplot(822)
plt.imshow(HI_matrix_test[1].reshape((32,32)), vmin = 0, vmax = maximum)
plt.colorbar()
plt.title('True Map of HI (901)')

plt.subplot(823)
hydrogen_pred2 = Model(DM_matrix_test[2]).detach().numpy()
maximum = np.max([np.max(hydrogen_pred2), torch.max(HI_matrix_test[2])])
plt.imshow(hydrogen_pred2.reshape((32,32)), vmin = 0, vmax = maximum)
plt.title('Prediction Map of HI (902)')

plt.subplot(824)
plt.imshow(HI_matrix_test[2].reshape((32,32)), vmin = 0, vmax = maximum)
plt.title('True Map of HI (902)')

plt.subplot(825)
hydrogen_pred3 = Model(DM_matrix_test[3]).detach().numpy() 
maximum = np.max([np.max(hydrogen_pred3), torch.max(HI_matrix_test[3])])
plt.imshow(hydrogen_pred3.reshape((32,32)), vmin = 0, vmax = maximum)
plt.title('Prediction Map of HI (903)')

plt.subplot(826)
plt.imshow(HI_matrix_test[3].reshape((32,32)), vmin = 0, vmax = maximum)
plt.title('True Map of HI (903)')

plt.subplot(827)
hydrogen_pred4 = Model(DM_matrix_test[4]).detach().numpy()
maximum = np.max([np.max(hydrogen_pred4), torch.max(HI_matrix_test[4])])
plt.imshow(hydrogen_pred4.reshape((32,32)), vmin = 0, vmax = maximum)
plt.title('Prediction Map of HI (904)')

plt.subplot(828)
plt.imshow(HI_matrix_test[4].reshape((32,32)), vmin = 0, vmax = maximum)
plt.title('True Map of HI (904)')
plt.subplots_adjust(bottom=10, right=1.5, top=17)
plt.show()


loss1 = criterion(hydrogen_pred1, HI_matrix_test[1])
print('Error 901: ', loss1.detach().numpy())
loss2 = criterion(hydrogen_pred2, HI_matrix_test[2])
print('Error 902: ', loss2.detach().numpy())
loss3 = criterion(hydrogen_pred3, HI_matrix_test[3])
print('Error 903: ', loss3.detach().numpy())
loss4 = criterion(hydrogen_pred4, HI_matrix_test[4])
print('Error 904: ', loss4.detach().numpy())

plt.subplot(821)
hydrogen_pred1 = Model(DM_matrix_test[1]).detach().numpy() 
maximum = np.max([np.max(hydrogen_pred1), torch.max(HI_matrix_test[1])])
plt.imshow(hydrogen_pred1.reshape((32,32)), vmin = 0, vmax = maximum)
plt.colorbar()
plt.title('Prediction Map of HI (901)')

plt.subplot(822)
plt.imshow(HI_matrix_test[1].reshape((32,32)), vmin = 0, vmax = maximum)
plt.colorbar()
plt.title('True Map of HI (901)')

plt.subplot(823)
hydrogen_pred2 = Model(DM_matrix_test[2]).detach().numpy()
maximum = np.max([np.max(hydrogen_pred2), torch.max(HI_matrix_test[2])])
plt.imshow(hydrogen_pred2.reshape((32,32)), vmin = 0, vmax = maximum)
plt.title('Prediction Map of HI (902)')

plt.subplot(824)
plt.imshow(HI_matrix_test[2].reshape((32,32)), vmin = 0, vmax = maximum)
plt.title('True Map of HI (902)')

plt.subplot(825)
hydrogen_pred3 = Model(DM_matrix_test[3]).detach().numpy() 
maximum = np.max([np.max(hydrogen_pred3), torch.max(HI_matrix_test[3])])
plt.imshow(hydrogen_pred3.reshape((32,32)), vmin = 0, vmax = maximum)
plt.title('Prediction Map of HI (903)')

plt.subplot(826)
plt.imshow(HI_matrix_test[3].reshape((32,32)), vmin = 0, vmax = maximum)
plt.title('True Map of HI (903)')

plt.subplot(827)
hydrogen_pred4 = Model(DM_matrix_test[4]).detach().numpy()
maximum = np.max([np.max(hydrogen_pred4), torch.max(HI_matrix_test[4])])
plt.imshow(hydrogen_pred4.reshape((32,32)), vmin = 0, vmax = maximum)
plt.title('Prediction Map of HI (904)')

plt.subplot(828)
plt.imshow(HI_matrix_test[4].reshape((32,32)), vmin = 0, vmax = maximum)
plt.title('True Map of HI (904)')
plt.subplots_adjust(bottom=10, right=1.5, top=17)
plt.show()


loss1 = criterion(hydrogen_pred1, HI_matrix_test[1])
print('Error 901: ', loss1.detach().numpy())
loss2 = criterion(hydrogen_pred2, HI_matrix_test[2])
print('Error 902: ', loss2.detach().numpy())
loss3 = criterion(hydrogen_pred3, HI_matrix_test[3])
print('Error 903: ', loss3.detach().numpy())
loss4 = criterion(hydrogen_pred4, HI_matrix_test[4])
print('Error 904: ', loss4.detach().numpy())
