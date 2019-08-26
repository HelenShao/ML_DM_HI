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
learning_rate = 0.001

#Find mean and standard deviation for data 

#Find mean and standard deviation for data sets

#Train Dataset
DM = np.zeros((1000, 1024), dtype = np.float32)
HI = np.zeros((1000, 1024), dtype = np.float32)

for i in range(1000):
    DM[i] = np.load('DM_new_map_%d.npy'%(i)).flatten()
    HI[i] = np.load('HI_new_map_%d.npy'%(i)).flatten()
            
    # Take the log
    DM[i] = np.log(DM[i] + 1)
    HI[i] = np.log(HI[i] + 1)
    
#Mean Values
DM_mean = np.mean(DM)
HI_mean = np.mean(HI)

#STD Values
DM_std = np.std(DM)
HI_std = np.std(HI)

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
            self.DM_matrix_train[i] = (self.DM_matrix_train[i] - DM_mean) / (DM_std)
            self.HI_matrix_train[i] = (self.HI_matrix_train[i] - HI_mean) / (HI_std)
        
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
            self.DM_matrix_valid[i] = (self.DM_matrix_valid[i] - DM_mean) / (DM_std)
            self.HI_matrix_valid[i] = (self.HI_matrix_valid[i] - HI_mean) / (HI_std)
        
        #Convert into torch tensor
        self.DM_matrix_valid = torch.tensor(self.DM_matrix_valid, dtype=torch.float)
        self.HI_matrix_valid = torch.tensor(self.HI_matrix_valid, dtype=torch.float)
            
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
    DM_matrix_test[i] = (DM_matrix_test[i] - DM_mean) / (DM_std)
    HI_matrix_test[i] = (HI_matrix_test[i] - HI_mean) / (HI_std)
        
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

#Call the Model

criterion = nn.MSELoss()       #Loss Function           
optimizer = torch.optim.Adam(Model.parameters(), lr = learning_rate)


##### Training Loop #####

Loss_total = np.zeros(3000) # Record Train Loss 
loss_valid = np.zeros(3000) # Record Validation Loss
loss_test = np.zeros(3000)  # Record Test Loss
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
    
    # Test Set
    count3 = 0 
    for i, (DM_matrix_test_new, HI_matrix_test_new) in enumerate(test_loader):
        Model.eval()  #Set model into eval mode to stop back prop
        HI_test = Model(DM_matrix_test_new)
        error_test = criterion(HI_test, HI_matrix_test_new)   #Loss for validation set
        partial_loss_test += error_test
        count3 += 1
    partial_loss_test = partial_loss_test / count3
    
    #Save Best Model 
    if loss_valid[epoch]<best_model:
        best_model = loss_valid[epoch]
        torch.save(Model.state_dict(), 'BestModelDM_HI_New_2.pt')

    #Print loss for both training and validation sets
    Loss_total[epoch] = partial_loss  
    loss_valid[epoch] = partial_loss_valid
    loss_test[epoch] = partial_loss_test
    print('Epoch:', epoch, 'Train_Loss: ', Loss_total[epoch], ' Valid_Error:', loss_valid[epoch], 'Test_Loss:', loss_test[epoch])
    

torch.save(Model.state_dict(), 'BestModelDM_HI_New_2.pt')

################### Loss Function #######################

# Plot loss as a function of epochs
epochs = np.arange(3000)

plt.plot(epochs, Loss_total, label = 'Loss_train')
plt.plot(epochs, loss_valid, label= 'Loss_Valid')
plt.plot(epochs, loss_test, label = 'Loss_Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

#################### Predictions #######################


if os.path.exists('BestModelDM_HI_New.pt'):
    Model.load_state_dict(torch.load('BestModelDM_HI_New.pt'))

Model.eval()

#Plot predictions versus true images (901-904)
plt.subplot(821)
hydrogen_pred1 = Model(DM_matrix_test[1])    #Make prediction
hydrogen_pred01 = hydrogen_pred1.detach().numpy()  #Detach variable for plotting
maximum = np.max([np.max(hydrogen_pred01), torch.max(HI_matrix_test[1])])
minimum = np.min([np.min(hydrogen_pred01), torch.min(HI_matrix_test[1])])
plt.imshow(hydrogen_pred01.reshape((32,32)), vmin = minimum, vmax = maximum)

plt.colorbar()
plt.title('Prediction Map of HI (901)')

plt.subplot(822)
plt.imshow(HI_matrix_test[1].reshape((32,32)), vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('True Map of HI (901)')

plt.subplot(823)
hydrogen_pred2 = Model(DM_matrix_test[2])
hydrogen_pred02 = hydrogen_pred2.detach().numpy()
maximum = np.max([np.max(hydrogen_pred02), torch.max(HI_matrix_test[2])])
minimum = np.min([np.min(hydrogen_pred02), torch.min(HI_matrix_test[2])])
plt.imshow(hydrogen_pred02.reshape((32,32)), vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('Prediction Map of HI (902)')

plt.subplot(824)
plt.imshow(HI_matrix_test[2].reshape((32,32)), vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('True Map of HI (902)')

plt.subplot(825)
hydrogen_pred3 = Model(DM_matrix_test[3])
hydrogen_pred03 = hydrogen_pred3.detach().numpy() 
maximum = np.max([np.max(hydrogen_pred03), torch.max(HI_matrix_test[3])])
minimum = np.min([np.min(hydrogen_pred03), torch.min(HI_matrix_test[3])])
plt.colorbar()
plt.imshow(hydrogen_pred03.reshape((32,32)), vmin = minimum, vmax = maximum)
plt.title('Prediction Map of HI (903)')

plt.subplot(826)
plt.imshow(HI_matrix_test[3].reshape((32,32)), vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('True Map of HI (903)')

plt.subplot(827)
hydrogen_pred4 = Model(DM_matrix_test[4])
hydrogen_pred04 = hydrogen_pred4.detach().numpy()
maximum = np.max([np.max(hydrogen_pred04), torch.max(HI_matrix_test[4])])
minimum = np.min([np.min(hydrogen_pred04), torch.min(HI_matrix_test[4])])
plt.colorbar()
plt.imshow(hydrogen_pred04.reshape((32,32)), vmin = minimum, vmax = maximum)
plt.title('Prediction Map of HI (904)')

plt.subplot(828)
plt.imshow(HI_matrix_test[4].reshape((32,32)), vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('True Map of HI (904)')

plt.subplots_adjust(bottom=10, right=1.5, top=17)
plt.show()

#Compute loss for predictions
loss1 = criterion(hydrogen_pred1, HI_matrix_test[1])
print('Error 901: ', loss1.detach().numpy())
loss2 = criterion(hydrogen_pred2, HI_matrix_test[2])
print('Error 902: ', loss2.detach().numpy())
loss3 = criterion(hydrogen_pred3, HI_matrix_test[3])
print('Error 903: ', loss3.detach().numpy())
loss4 = criterion(hydrogen_pred4, HI_matrix_test[4])
print('Error 904: ', loss4.detach().numpy())

# Plot the Predictions versus true images(905-908)
plt.subplot(821)
hydrogen_pred5 = Model(DM_matrix_test[5])    #Make prediction
hydrogen_pred05 = hydrogen_pred1.detach().numpy()  #Detach variable for plotting
maximum = np.max([np.max(hydrogen_pred05), torch.max(HI_matrix_test[5])])
minimum = np.min([np.min(hydrogen_pred05), torch.min(HI_matrix_test[5])])
plt.imshow(hydrogen_pred01.reshape((32,32)), vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('Prediction Map of HI (905)')

plt.subplot(822)
plt.imshow(HI_matrix_test[5].reshape((32,32)), vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('True Map of HI (905)')

plt.subplot(823)
hydrogen_pred6 = Model(DM_matrix_test[6])
hydrogen_pred06 = hydrogen_pred6.detach().numpy()
maximum = np.max([np.max(hydrogen_pred06), torch.max(HI_matrix_test[6])])
minimum = np.min([np.min(hydrogen_pred06), torch.min(HI_matrix_test[6])])
plt.imshow(hydrogen_pred06.reshape((32,32)), vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('Prediction Map of HI (906)')

plt.subplot(824)
plt.imshow(HI_matrix_test[6].reshape((32,32)), vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('True Map of HI (906)')

plt.subplot(825)
hydrogen_pred7 = Model(DM_matrix_test[7])
hydrogen_pred07 = hydrogen_pred7.detach().numpy() 
maximum = np.max([np.max(hydrogen_pred07), torch.max(HI_matrix_test[7])])
minimum = np.min([np.min(hydrogen_pred07), torch.min(HI_matrix_test[7])])
plt.colorbar()
plt.imshow(hydrogen_pred07.reshape((32,32)), vmin = minimum, vmax = maximum)
plt.title('Prediction Map of HI (907)')

plt.subplot(826)
plt.imshow(HI_matrix_test[7].reshape((32,32)), vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('True Map of HI (907)')

plt.subplot(827)
hydrogen_pred8 = Model(DM_matrix_test[8])
hydrogen_pred08 = hydrogen_pred4.detach().numpy()
maximum = np.max([np.max(hydrogen_pred08), torch.max(HI_matrix_test[8])])
minimum = np.min([np.min(hydrogen_pred08), torch.min(HI_matrix_test[8])])
plt.colorbar()
plt.imshow(hydrogen_pred08.reshape((32,32)), vmin = minimum, vmax = maximum)
plt.title('Prediction Map of HI (908)')

plt.subplot(828)
plt.imshow(HI_matrix_test[8].reshape((32,32)), vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('True Map of HI (908)')

plt.subplots_adjust(bottom=10, right=1.5, top=17)
plt.show()

loss5 = criterion(hydrogen_pred5, HI_matrix_test[5])
print('Error 905: ', loss5.detach().numpy())
loss6 = criterion(hydrogen_pred6, HI_matrix_test[6])
print('Error 906: ', loss6.detach().numpy())
loss7 = criterion(hydrogen_pred7, HI_matrix_test[7])
print('Error 907: ', loss7.detach().numpy())
loss8 = criterion(hydrogen_pred8, HI_matrix_test[8])
print('Error 908: ', loss8.detach().numpy())
