import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import sys, os
import matplotlib.pyplot as plt

###################################################################
# This function implements the transformation we do to the data
def transform(x):
    return np.log10(x+1.0) #x**0.25
###################################################################

###################################################################
# This function reads all data, normalizes it, and adds Gaussian Noise
def read_all_data(maps, normalize=True, noise=False):

    # Load estimated HI (input) maps
    HI_esti = np.load('Esti_transform_large.npy')

    # Load HI and DM (True) maps 
    DM_True = np.load('DM_transform_large.npy')
    HI_True = np.load('HI_transform_large.npy')
    
    # Take the log
    DM_True = transform(DM_True)
    HI_True = transform(HI_True)
  
    # normalize data
    if normalize:
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

    return DM_True, HI_True, HI_esti
###################################################################

###################################################################
# Define Dataset 
class make_Dataset(Dataset):
    
    def __init__(self, name, seed, maps, DM0, HI0, HI1):
         
        # shuffle the maps (instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342)
        np.random.seed(seed)
        indexes = np.arange(maps)
        np.random.shuffle(indexes)

        if   name=='train':  size, offset = int(maps*0.8), int(maps*0.0)
        elif name=='valid':  size, offset = int(maps*0.1), int(maps*0.8)
        elif name=='test':   size, offset = int(maps*0.1), int(maps*0.9)
        else:                raise Exception('Wrong name!')
        
        self.size   = size
        self.input  = torch.zeros((size, 2, 32, 32), dtype=torch.float)
        self.output = torch.zeros((size, 1, 32, 32), dtype=torch.float)
        
        # do a loop over all elements in the dataset
        for i in range(size):
            
            # find the index of the map
            j = indexes[i+offset]

            # load maps
            self.input [i,0,:,:] = DM0[j]
            self.input [i,1,:,:] = HI1[j]
            self.output[i,0,:,:] = HI0[j]
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]
###################################################################

###################################################################
# This function creates the different datasets
def create_datasets(name, seed, maps, DM0, HI0, HI1, batch_size):

    data_set     = make_Dataset(name, seed, maps, DM0, HI0, HI1)
    data_loader  = DataLoader(dataset=data_set, batch_size=batch_size, 
                               shuffle=True)
    return data_loader
###################################################################

###################################################################
# This function creates the test set for making images
def create_testset_images(seed, maps, DM0, HI0, HI1):

    test_Dataset  = make_Dataset('test', seed, maps, DM0, HI0, HI1)
    test_loader   = DataLoader(dataset=test_Dataset,  batch_size=1, 
                               shuffle=False)

    return test_loader
###################################################################


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

class UNet2(nn.Module):

    def __init__(self):
        super(UNet2,self).__init__()
                
        self.dconv_down1 = double_conv(2, 4)
        self.dconv_down2 = double_conv(4, 8)
        self.dconv_down3 = double_conv(8, 16)
        self.dconv_down4 = double_conv(16, 32)        

        self.maxpool = nn.MaxPool2d(2)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', 
                                    align_corners=True)        
        
        self.dconv_up3 = double_conv(32 + 16, 32)     #concatenated with conv_down3 
        self.dconv_up2 = double_conv(32 + 8, 16)      #concatenated with conv_down2
        self.dconv_up1 = double_conv(16 + 2 + 4, 8)   #concatenated with conv_down1 and 2 inputs
        self.conv_last = nn.Conv2d(8, 1, 1) 
        
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
        
        return out
    


################################ INPUT #######################################

maps = 98304
seed = 42   #random seed to split maps in train, valid and test sets

# Define NN hyperparameters
num_epochs    = 500
batch_size    = 64
learning_rate = 0.00003
weight_decay  = 1e-1

f_best_model = 'UNet_First.pt'

##############################################################################

# use GPUs 
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print('GPU: %s     ||    Training on %s\n'%(GPU,device))
#cudnn.benchmark = True      #Trains faster but costs more memory

################ READ DATA ###################
# read all data
DM_True, HI_True, HI_esti = read_all_data(maps, normalize=True, noise=False)

# create training, validation and test datasets and data_loaders
train_loader = create_datasets('train', seed, maps, DM_True, HI_True, HI_esti, batch_size)
valid_loader = create_datasets('valid', seed, maps, DM_True, HI_True, HI_esti, batch_size)
test_loader  = create_datasets('test' , seed, maps, DM_True, HI_True, HI_esti, batch_size)

##############################################


########## MODEL, LOSS and OPTIMIZER #########
# define architecture
model = UNet2().to(device)

# Create loss and optimizer function for training  
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                             weight_decay=weight_decay)
##############################################

#Record loss in array for plotting
Loss_train = np.zeros(500)
Loss_valid = np.zeros(500)
Loss_test  = np.zeros(500)

############## LOAD BEST-MODEL ###############
#best_loss = 1.6e-1
if os.path.exists(f_best_model):
    model.load_state_dict(torch.load(f_best_model))

    # do validation with the best-model and compute loss
    model.eval() 
    count, best_loss = 0, 0.0
    for input, HI_True in valid_loader:
        input = input.to(device=device) #valid_best
        HI_True = HI_True.to(device=device)
        HI_valid = model(input)
        error    = criterion(HI_valid, HI_True)
        best_loss += error.cpu().detach().numpy()
        count += 1
    best_loss /= count
    print('validation error = %.3e'%best_loss)
##############################################


############## TRAIN AND VALIDATE ############
for epoch in range(num_epochs):

    # TRAIN
    model.train()
    count, loss_train = 0, 0.0
    for input, HI_True in train_loader:        
        # Forward Pass
        #input = (input+torch.randn(32, 32))*((torch.mean(input)+torch.std(input))*0.5)
        input = input.to(device)  #train
        HI_True = HI_True.to(device)
        HI_pred = model(input)
        loss    = criterion(HI_pred, HI_True)
        loss_train += loss.cpu().detach().numpy()
        #print(torch.mean(input_noise), torch.std(input_noise))
        
        # Backward Prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        count += 1
    loss_train /= count
    Loss_train[epoch] = loss_train
    
    # VALID
    model.eval() 
    count, loss_valid = 0, 0.0
    for input, HI_True in valid_loader:
        input = input.to(device) #valid
        HI_True = HI_True.to(device)
        HI_valid = model(input)
        error    = criterion(HI_valid, HI_True)   
        loss_valid += error.cpu().detach().numpy()
        count += 1
    loss_valid /= count
    Loss_valid[epoch] = loss_valid
    
    # TEST
    model.eval() 
    count, loss_test = 0, 0.0
    for input, HI_True in test_loader:
        input = input.to(device) #test
        HI_True = HI_True.to(device)
        HI_test  = model(input)
        error    = criterion(HI_test, HI_True) 
        loss_test += error.cpu().detach().numpy()
        count += 1
    loss_test /= count
    Loss_test[epoch] = loss_test
    
    # Save Best Model 
    if loss_valid<best_loss:
        best_loss = loss_valid
        torch.save(model.state_dict(), f_best_model)
        print('%03d %.4e %.4e %.4e (saving)'\
              %(epoch, loss_train, loss_valid, loss_test))
    else:
        print('%03d %.4e %.4e %.4e'%(epoch, loss_train, loss_valid, loss_test))


################### Loss Function #######################
# Plot loss as a function of epochs
import matplotlib.pyplot as plt
epochs = np.arange(num_epochs)
plt.plot(epochs, Loss_train, label = 'Loss_train')
plt.plot(epochs, Loss_valid, label= 'Loss_Valid')
plt.plot(epochs, Loss_test, label = 'Loss_Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
