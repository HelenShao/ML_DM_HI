import numpy as np
import matplotlib.pyplot as plt

#Load the full maps
HI_Full = np.load('HI_new_full_map_1.npy').astype('float32')  
DM_Full = np.load('DM_new_full_map_1.npy').astype('float32')
HI_estimated = np.load('HI_estimated.npy').astype('float32')

#Slice the Maps into 4096 (32x32) images
for i in range(64):
    for j in range(64):
        HI_True = HI_Full[32*i:32+(32*i), (32*j):32*(j+1)]  #HI Maps
        np.save('HI_map_%d_%d.npy'%(i,j), HI_True)

        DM_True = DM_Full[32*i:32+(32*i), (32*j):32*(j+1)]  #DM Maps
        np.save('DM_map_%d_%d.npy'%(i,j), DM_True)

        HI_esti = HI_estimated[32*i:32+(32*i), (32*j):32*(j+1)]  #HI Estimated Maps
        np.save('Esti_map_%d_%d.npy'%(i,j), HI_esti)
        

