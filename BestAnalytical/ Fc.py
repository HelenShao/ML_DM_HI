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
maps = 10000
seed = 6    #random seed to split maps in train, valid and test sets

# Define NN hyperparameters
num_epochs    = 500
batch_size    = 16
learning_rate = 0.0001

f_best_model = 'BestModelDM_HI_log.pt'
##############################################################################


# Find mean and std of all maps
DM = np.zeros((maps, 1024), dtype=np.float32)
HI = np.zeros((maps, 1024), dtype=np.float32)

# do a loop over all maps
for i in range(maps):
    DM[i] = np.load('../maps/DM_new_map_%d.npy'%(i)).flatten()
    HI[i] = np.load('../maps/HI_new_map_%d.npy'%(i)).flatten()
            
    # Take the log
    DM[i], HI[i] = transform(DM[i]), transform(HI[i])
    #DM[i], HI[i] = np.log(DM[i] + 1.0), np.log(HI[i] + 1.0)
    
# Mean and std values
DM_mean, DM_std = np.mean(DM), np.std(DM)
HI_mean, HI_std = np.mean(HI), np.std(HI)
print DM_mean, DM_std
print HI_mean, HI_std

# normalize matrices
DM = (DM - DM_mean)/DM_std
HI = (HI - HI_mean)/HI_std

############# Create Datasets ############
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
        self.DM_matrix = np.zeros((size, 1024), dtype=np.float32)
        self.HI_matrix = np.zeros((size, 1024), dtype=np.float32)
        
        for i in range(size):
            
            # find the index of the map
            j = indexes[i+offset]

            # read maps
            self.DM_matrix[i] = np.load('../maps/DM_new_map_%d.npy'%j).flatten()
            self.HI_matrix[i] = np.load('../maps/HI_new_map_%d.npy'%j).flatten()
            
            # Transform data
            self.DM_matrix[i] = transform(self.DM_matrix[i])
            self.HI_matrix[i] = transform(self.HI_matrix[i])
            #self.DM_matrix[i] = np.log(self.DM_matrix[i] + 1.0)
            #self.HI_matrix[i] = np.log(self.HI_matrix[i] + 1.0)
            
            # Normalize: (x-mean)/(std)
            self.DM_matrix[i] = (self.DM_matrix[i] - DM_mean) / DM_std
            self.HI_matrix[i] = (self.HI_matrix[i] - HI_mean) / HI_std
        
        # Convert into torch tensor
        self.DM_matrix = torch.tensor(self.DM_matrix, dtype=torch.float)
        self.HI_matrix = torch.tensor(self.HI_matrix, dtype=torch.float)
            
    def __len__(self):
            return self.size

    def __getitem__(self, idx):
        return self.DM_matrix[idx], self.HI_matrix[idx]

train_Dataset = make_Dataset('train', seed)
train_loader  = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle=True)

valid_Dataset = make_Dataset('valid', seed)
valid_loader  = DataLoader(dataset=valid_Dataset, batch_size=batch_size, shuffle=True)

test_Dataset  = make_Dataset('test',  seed)
test_loader   = DataLoader(dataset=test_Dataset,  batch_size=batch_size, shuffle=True)




# Create Model Container: 3 fully connected layers (linear)
Model = nn.Sequential()

# Define Layers
fc1  = nn.Linear(1024,1200)      #Input size is the size of channels for the DM maps
fc2  = nn.Linear(1200,1500)      
fc3  = nn.Linear(1500,1500)      
fc4  = nn.Linear(1500,1200)      
fc5  = nn.Linear(1200,1024)      #Output size is the size of channels for HI maps
Relu = nn.LeakyReLU()
Dropout = nn.Dropout(p=0.5)

# Add layers to Model
Model.add_module('Lin1',  fc1)
Model.add_module('DP1',   Dropout)
Model.add_module('Relu1', Relu)
Model.add_module('fc2',   fc2)
Model.add_module('DP2',   Dropout)
Model.add_module('Relu2', Relu)
Model.add_module('fc3',   fc3)
Model.add_module('DP3',   Dropout)
Model.add_module('Relu3', Relu)
Model.add_module('fc4',   fc4)
Model.add_module('DP4',   Dropout)
Model.add_module('Relu4', Relu)
Model.add_module('fc5',   fc5)

# Call the Model
criterion = nn.MSELoss()       #Loss Function           
optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate, weight_decay=0.0)


##### Training Loop #####
loss_train = np.zeros(3000) #Record Train Loss 
loss_valid = np.zeros(3000) #Record Validation Loss
loss_test  = np.zeros(3000) #Record Test Loss
best_model = 9e7

# load best-model
if os.path.exists(f_best_model):  
    Model.load_state_dict(torch.load(f_best_model))

# do a loop over the different epochs
for epoch in range(num_epochs):
    
    """
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
    """
            
    count = 0
    partial_loss_train = 0.0
    for DM_train, HI_train in train_loader:
        # Forward Pass
        Model.train()
        HI_pred = Model(DM_train)
        loss = criterion(HI_pred, HI_train)   #Loss  for train set
        partial_loss_train += loss.detach().numpy()
        # Backward Prop
        optimizer.zero_grad();  loss.backward();  optimizer.step()
        count += 1
    partial_loss_train /= count  #Divide partial_loss by number of iterations
       
    # Validation Set
    count2, partial_loss_valid = 0, 0.0
    for DM_valid, HI_valid in valid_loader:
        Model.eval()  #Set model into eval mode to stop back prop
        HI_pred = Model(DM_valid)
        error = criterion(HI_pred, HI_valid)   #Loss for validation set=
        partial_loss_valid += error
        count2 += 1
    partial_loss_valid /= count2
    
    # Test Set
    count3, partial_loss_test = 0, 0.0
    for DM_test, HI_test in test_loader:
        Model.eval()  #Set model into eval mode to stop back prop
        HI_pred = Model(DM_test)
        error = criterion(HI_pred, HI_test)   #Loss for validation set
        partial_loss_test += error
        count3 += 1
    partial_loss_test /= count3
    
    # Save Best Model 
    if partial_loss_valid<best_model:
        best_model = partial_loss_valid
        torch.save(Model.state_dict(), f_best_model)

    # Print loss for both training and validation sets
    loss_train[epoch] = partial_loss_train
    loss_valid[epoch] = partial_loss_valid
    loss_test[epoch]  = partial_loss_test
    print('Epoch %04d ----> Train = %.3f ----> Valid = %.3f ----> Test = %.3f'\
          %(epoch, loss_train[epoch], loss_valid[epoch], loss_test[epoch]))

    


"""
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
"""


#################### Predictions #######################
np.random.seed(seed)
indexes = np.arange(maps)
np.random.shuffle(indexes)

# make DM a torch tensor for predictions
DM = torch.tensor(DM, dtype=torch.float)

if os.path.exists(f_best_model): 
    Model.load_state_dict(torch.load(f_best_model))
Model.eval()

# maps for 9001
Map_in  = DM[indexes[9001]]
Map_out = HI[indexes[9001]]
plt.subplot(421)
Map_pred = Model(Map_in).detach().numpy()    #Make prediction
maximum = np.max([np.max(Map_pred), np.max(Map_out)])
minimum = np.min([np.min(Map_pred), np.min(Map_out)])
plt.imshow(Map_pred.reshape((32,32)), vmin=minimum, vmax=maximum)
plt.colorbar();  plt.title('Prediction Map of HI (901)')
plt.subplot(422)
plt.imshow(Map_out.reshape((32,32)), vmin=minimum, vmax=maximum)
plt.colorbar();  plt.title('True Map of HI (901)')
loss = np.mean((Map_pred-Map_out)**2)
print 'loss1 = %.3f'%loss

# maps for 9002
Map_in  = DM[indexes[9002]]
Map_out = HI[indexes[9002]]
plt.subplot(423)
Map_pred = Model(Map_in).detach().numpy()    #Make prediction
maximum = np.max([np.max(Map_pred), np.max(Map_out)])
minimum = np.min([np.min(Map_pred), np.min(Map_out)])
plt.imshow(Map_pred.reshape((32,32)), vmin=minimum, vmax=maximum)
plt.colorbar();  plt.title('Prediction Map of HI (902)')
plt.subplot(424)
plt.imshow(Map_out.reshape((32,32)), vmin=minimum, vmax=maximum)
plt.colorbar();  plt.title('True Map of HI (902)')
loss = np.mean((Map_pred-Map_out)**2)
print 'loss1 = %.3f'%loss

# maps for 9003
Map_in  = DM[indexes[9003]]
Map_out = HI[indexes[9003]]
plt.subplot(425)
Map_pred = Model(Map_in).detach().numpy()    #Make prediction
maximum = np.max([np.max(Map_pred), np.max(Map_out)])
minimum = np.min([np.min(Map_pred), np.min(Map_out)])
plt.imshow(Map_pred.reshape((32,32)), vmin=minimum, vmax=maximum)
plt.colorbar();  plt.title('Prediction Map of HI (903)')
plt.subplot(426)
plt.imshow(Map_out.reshape((32,32)), vmin=minimum, vmax=maximum)
plt.colorbar();  plt.title('True Map of HI (903)')
loss = np.mean((Map_pred-Map_out)**2)
print 'loss1 = %.3f'%loss

# maps for 9004
Map_in  = DM[indexes[9004]]
Map_out = HI[indexes[9004]]
plt.subplot(427)
Map_pred = Model(Map_in).detach().numpy()    #Make prediction
maximum = np.max([np.max(Map_pred), np.max(Map_out)])
minimum = np.min([np.min(Map_pred), np.min(Map_out)])
plt.imshow(Map_pred.reshape((32,32)), vmin=minimum, vmax=maximum)
plt.colorbar();  plt.title('Prediction Map of HI (904)')
plt.subplot(428)
plt.imshow(Map_out.reshape((32,32)), vmin=minimum, vmax=maximum)
plt.colorbar();  plt.title('True Map of HI (904)')
loss = np.mean((Map_pred-Map_out)**2)
print 'loss1 = %.3f'%loss

#plt.subplots_adjust(bottom=10, right=1.5, top=17)
plt.show()

# Compute loss for predictions
#loss1 = criterion(hydrogen_pred1, HI_matrix_test[1])
#print('Error 901: ', loss1.detach().numpy())
#loss2 = criterion(hydrogen_pred2, HI_matrix_test[2])
#print('Error 902: ', loss2.detach().numpy())
#loss3 = criterion(hydrogen_pred3, HI_matrix_test[3])
#print('Error 903: ', loss3.detach().numpy())
#loss4 = criterion(hydrogen_pred4, HI_matrix_test[4])
#print('Error 904: ', loss4.detach().numpy())



