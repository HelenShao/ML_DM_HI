import torch
import torchvision
import torch.nn as nn
import sys, os
import numpy as np
import matplotlib.pyplot as plt

################################ INPUT #######################################
root = '../maps'

maps = 8000
seed = 124    #random seed to split maps in train, valid and test sets

# Define NN hyperparameters
num_epochs    = 100
batch_size    = 16
learning_rate = 0.0001

f_best_model = 'BestModel_UNET3.pt'
##############################################################################


################ READ DATA ###################
# read all data
DM_True, HI_True, HI_esti = read_all_data(maps, root, normalize=True)

# create training, validation and test datasets and data_loaders
train_loader, valid_loader, test_loader = create_datasets(seed, 
                                maps, DM_True, HI_True, HI_esti, batch_size)
##############################################


########## MODEL, LOSS and OPTIMIZER #########
# define architecture
model = UNet2()

# Create loss and optimizer function for training  
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                             weight_decay=0.00001)
##############################################

#Record loss in array for plotting
Loss_train = np.zeros(100)
Loss_valid = np.zeros(100)
Loss_test  = np.zeros(100)

############## LOAD BEST-MODEL ###############
best_loss = 3.7e-1
if os.path.exists(f_best_model):
    model.load_state_dict(torch.load(f_best_model))

    # do validation with the best-model and compute loss
    model.eval() 
    count, best_model = 0, 0.0
    for input, HI_True in valid_loader:
        HI_valid   = model(input)
        error      = criterion(HI_valid, HI_True)
        best_loss += error.detach().numpy()
        count += 1
    best_loss /= count
    print('validation error = %.3e'%best_loss)
##############################################


############## TRAIN AND VALIDATE ############
for epoch in range(num_epochs):

    # TRAIN
    model.train()
    count, loss_train = 0, 0.0
    for input, HI_True_train in train_loader:        
        # Forward Pass
        HI_pred = model(input)
        loss    = criterion(HI_pred, HI_True_train)
        loss_train += loss.detach().numpy()
        
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
    for input, HI_True_valid in valid_loader:
        HI_valid = model(input)
        error    = criterion(HI_valid, HI_True_valid)   
        loss_valid += error.detach().numpy()
        count += 1
    loss_valid /= count
    Loss_valid[epoch] = loss_valid
    
    # TEST
    model.eval() 
    count, loss_test = 0, 0.0
    for input, HI_True_test in test_loader:
        HI_test  = model(input)
        error    = criterion(HI_test, HI_True_test) 
        loss_test += error.detach().numpy()
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
    
    # save results to file
    #f = open('results_UNET3.txt', 'a')
    #f.write('%d %.4e %.4e %.4e\n'%(epoch, loss_train, loss_valid, loss_test))
    #f.close()
