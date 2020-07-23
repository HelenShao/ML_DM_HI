import numpy as np
import matplotlib.pyplot as plt

# Load Maps
DM= np.load('DM_maps_new.npy')
HI= np.load('HI_maps_new.npy')

# Create Histogram
bins_DM = np.linspace(0,28,50)      #0/28 are the min/max values of the DM maps
bins_DM_mean = 0.5*(bins_DM[1:] + bins_DM[:-1] )     
Number_points = np.histogram(DM, bins=bins_DM)[0]
HI_total      = np.histogram(DM, bins=bins_DM, weights=HI)[0]
HI_mean = HI_total / Number_points
HI_mean = np.nan_to_num(HI_mean)    #Change NaN values in array to 0
    
# Estimate HI using the best fit line and load into HI_estimation matrix 
HI_estimation = np.interp(DM, bins_DM_mean, HI_mean)
np.save('HI_estimated_large.npy', HI_estimation)
