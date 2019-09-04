import numpy as np
import torch
import data as data
import Unet_architecture as architecture
import sys,os

from pylab import *
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
rcParams["mathtext.fontset"]='cm'

################################# INPUT #######################################
root          = '/mnt/ceph/users/fvillaescusa/Helen/maps'
f_best_model1 = 'BestModel_UNET_FULL_1e-3.pt'
f_best_model2 = 'BestModel_UNET_FULL_RF_1e-3.pt'
maps          = 15000
seed          = 6
###############################################################################

################## MODEL ####################
model1 = architecture.UNet()
model2 = architecture.UNet()

model1.load_state_dict(torch.load(f_best_model1))
model2.load_state_dict(torch.load(f_best_model2))
    
model1.eval()
model2.eval()
##############################################

################ READ DATA ###################
# read all data
DM_True, HI_True, HI_esti, HI_RF = \
        data.read_all_data(maps, root, normalize=True, torch_tensors=True)
##############################################


"""
count = 0
losses = np.zeros(1000, dtype=np.float32)
for Map_in, Map_True in test_loader:
    Map_pred = model2(Map_in).detach().numpy() 
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
sys.exit()
"""


np.random.seed(seed)
indexes = np.arange(maps)
np.random.shuffle(indexes)


x_min, x_max = 0.0, 1.17
y_min, y_max = 0.0, 1.17

fs = 12 #fontsize

# do a loop over the first 100 test maps
for i in xrange(100):

    # find the index of the map
    j = indexes[9000+i]

    f_out = 'new_results/Test_%d.pdf'%i

    fig = figure(figsize=(30,8))
    ax1 = fig.add_subplot(241) 
    ax2 = fig.add_subplot(242) 
    ax3 = fig.add_subplot(243) 
    ax4 = fig.add_subplot(244) 
    ax5 = fig.add_subplot(245) 
    ax6 = fig.add_subplot(246) 
    ax7 = fig.add_subplot(247) 
    ax8 = fig.add_subplot(248) 
    
    for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]:
        ax.set_xlabel(r'$X\/[h^{-1}\/{\rm Mpc}]$',fontsize=fs)
        ax.set_ylabel(r'$Y\/[h^{-1}\/{\rm Mpc}]$',fontsize=fs)

    # read data and make predictions
    Map_in1 = torch.cat([DM_True[j].unsqueeze(0), 
                         HI_esti[j].unsqueeze(0)], dim=0).unsqueeze(0)
    Map_in2 = torch.cat([DM_True[j].unsqueeze(0), 
                         HI_RF[j].unsqueeze(0)], dim=0).unsqueeze(0)

    HI_pred1 = model1(Map_in1)[0,0,:,:].detach().numpy() 
    HI_pred2 = model2(Map_in2)[0,0,:,:].detach().numpy() 

    # get DM and HI maps
    DM = DM_True[j].numpy()
    DM_min,DM_max,DM_mean,DM_std = np.min(DM),np.max(DM),np.mean(DM),np.std(DM)
    HI0 = HI_True[j].numpy()
    HI1 = HI_esti[j].numpy()
    HI2 = HI_RF[j].numpy()
    HI_min = np.min([HI0, HI1, HI2, HI_pred1, HI_pred2])
    HI_max = np.max([HI0, HI1, HI2, HI_pred1, HI_pred2])

    # DM
    for ax in [ax1,ax5]:
        cax = ax.imshow(DM, vmin=DM_mean-3*DM_std, vmax=DM_mean+3*DM_std, 
                        origin='lower',
                        extent=[x_min,x_max,y_min,y_max], cmap=get_cmap('hot'))
        cbar = fig.colorbar(cax, ax=ax) 
        cbar.set_label(r"${\rm log}(1+\delta_{\rm DM})$",fontsize=fs,labelpad=0)
        ax.set_title('DM', fontsize=fs)

    # HI estimation
    cax = ax2.imshow(HI1, vmin=HI_min, vmax=HI_max, origin='lower',
                     extent=[x_min,x_max,y_min,y_max], cmap=get_cmap('jet'))
    cbar = fig.colorbar(cax, ax=ax2) 
    cbar.set_label(r"${\rm log}(1+\delta_{\rm HI})$",fontsize=fs,labelpad=0)
    ax2.set_title('HI estimation', fontsize=fs)

    # HI RF
    cax = ax6.imshow(HI2, vmin=HI_min, vmax=HI_max, origin='lower',
                     extent=[x_min,x_max,y_min,y_max], cmap=get_cmap('jet'))
    cbar = fig.colorbar(cax, ax=ax6) 
    cbar.set_label(r"${\rm log}(1+\delta_{\rm HI})$",fontsize=fs,labelpad=0)
    ax6.set_title('HI random forest', fontsize=fs)

    # HI pred from estimation
    cax = ax3.imshow(HI_pred1, vmin=HI_min, vmax=HI_max, origin='lower',
                     extent=[x_min,x_max,y_min,y_max], cmap=get_cmap('jet'))
    cbar = fig.colorbar(cax, ax=ax3) 
    cbar.set_label(r"${\rm log}(1+\delta_{\rm HI})$",fontsize=fs,labelpad=0)
    ax3.set_title('HI prediction', fontsize=fs)

    # HI pred from random forest
    cax = ax7.imshow(HI_pred2, vmin=HI_min, vmax=HI_max, origin='lower',
                     extent=[x_min,x_max,y_min,y_max], cmap=get_cmap('jet'))
    cbar = fig.colorbar(cax, ax=ax7) 
    cbar.set_label(r"${\rm log}(1+\delta_{\rm HI})$",fontsize=fs,labelpad=0)
    ax7.set_title('HI prediction', fontsize=fs)

    # HI True
    for ax in [ax4,ax8]:
        cax = ax.imshow(HI0, vmin=HI_min, vmax=HI_max, origin='lower',
                        extent=[x_min,x_max,y_min,y_max], cmap=get_cmap('jet'))
        cbar = fig.colorbar(cax, ax=ax) 
        cbar.set_label(r"${\rm log}(1+\delta_{\rm HI})$",fontsize=fs,labelpad=0)
        ax.set_title('HI True', fontsize=fs)

    loss1 = np.mean((HI_pred1 - HI0)**2)
    loss2 = np.mean((HI_pred2 - HI0)**2)
    print('loss1 = %.3f  :  loss2 = %.3f'%(loss1,loss2))

    suptitle('loss1 = %.3f  :  loss2 = %.3f'%(loss1,loss2), fontsize=fs)
    savefig(f_out, bbox_inches='tight')
    close(fig)


