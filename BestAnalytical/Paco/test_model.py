import numpy as np
import torch
import matplotlib.pyplot as plt
import data as data
import Unet_architecture as architecture
import sys,os

################################# INPUT #######################################
root         = '/mnt/ceph/users/fvillaescusa/Helen/maps'
f_best_model = 'BestModel_UNET_A.pt'
maps         = 10000
seed         = 6
###############################################################################

################## MODEL ####################
model = architecture.UNet()

if not(os.path.exists(f_best_model)): raise Exception('File doesnt exists!!!') 
model.load_state_dict(torch.load(f_best_model))
    
model.eval()
##############################################

################ READ DATA ###################
# read all data
DM_True, HI_True, HI_esti = data.read_all_data(maps, root, normalize=True)
test_loader = data.create_testset_images(seed, maps, DM_True, HI_True, HI_esti)
##############################################

#data = iter(test_loader)

count = 0
losses = np.zeros(1000, dtype=np.float32)
for Map_in, Map_True in test_loader:
    Map_pred = model(Map_in).detach().numpy() 
    Map_True = Map_True.numpy()
    loss = np.mean((Map_pred - Map_True)**2)
    print('%03d ---> loss = %.3f'%(count,loss))
    losses[count] = loss
    count += 1
print np.min(losses), np.max(losses), np.mean(losses)

bins_histo = np.linspace(0, np.max(losses), 1000)
mean_bins_histo = 0.5*(bins_histo[:1] + bins_histo[:-1])
histo = np.histogram(losses, bins_histo)[0]
plt.subplot(111)
plt.xscale('log')
plt.yscale('log')
plt.plot(mean_bins_histo,histo)
plt.show()




# Maps for 9001
Map_in, Map_True = next(data)
plt.subplot(821)
Map_pred = model(Map_in).detach().numpy() 
Map_True = Map_True.numpy()
maximum  = np.max([np.max(Map_pred), np.max(Map_True)])
minimum  = np.min([np.min(Map_pred), np.min(Map_True)])
plt.imshow(Map_pred[0,0,:,:], vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('Prediction')
plt.subplot(822)
plt.imshow(Map_True[0,0,:,:], vmin=minimum, vmax=maximum)
plt.colorbar()
plt.title('True')
loss = np.mean((Map_pred - Map_True)**2)
print('loss1 = %.3f'%loss)

# Maps for 9002
Map_in, Map_True = next(data)
plt.subplot(823)
Map_pred = model(Map_in).detach().numpy() 
Map_True = Map_True.numpy()
maximum  = np.max([np.max(Map_pred), np.max(Map_True)])
minimum  = np.min([np.min(Map_pred), np.min(Map_True)])
plt.imshow(Map_pred[0,0,:,:], vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('Prediction')
plt.subplot(824)
plt.imshow(Map_True[0,0,:,:], vmin=minimum, vmax=maximum)
plt.colorbar()
plt.title('True')
loss = np.mean((Map_pred - Map_True)**2)
print('loss1 = %.3f'%loss)

# Maps for 9003
Map_in, Map_True = next(data)
plt.subplot(825)
Map_pred = model(Map_in).detach().numpy() 
Map_True = Map_True.numpy()
maximum  = np.max([np.max(Map_pred), np.max(Map_True)])
minimum  = np.min([np.min(Map_pred), np.min(Map_True)])
plt.imshow(Map_pred[0,0,:,:], vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('Prediction')
plt.subplot(826)
plt.imshow(Map_True[0,0,:,:], vmin=minimum, vmax=maximum)
plt.colorbar()
plt.title('True')
loss = np.mean((Map_pred - Map_True)**2)
print('loss1 = %.3f'%loss)

# Maps for 9004
Map_in, Map_True = next(data)
plt.subplot(827)
Map_pred = model(Map_in).detach().numpy() 
Map_True = Map_True.numpy()
maximum  = np.max([np.max(Map_pred), np.max(Map_True)])
minimum  = np.min([np.min(Map_pred), np.min(Map_True)])
plt.imshow(Map_pred[0,0,:,:], vmin = minimum, vmax = maximum)
plt.colorbar()
plt.title('Prediction')
plt.subplot(828)
plt.imshow(Map_True[0,0,:,:], vmin=minimum, vmax=maximum)
plt.colorbar()
plt.title('True')
loss = np.mean((Map_pred - Map_True)**2)
print('loss1 = %.3f'%loss)


#plt.subplots_adjust(bottom=10, right=1.5, top=17)
plt.show()

