import numpy as np
import matplotlib.pyplot as plt

#Create arrays to load maps
DM = np.zeros((1000, 1024))
HI = np.zeros((1000, 1024))

#Load maps
for i in range(1000):
    DM[i] = (np.log(np.load('DM_new_map_%d.npy'%(i))+1)).flatten()
    HI[i] = (np.log(np.load('HI_map_%d.npy'%(i))+1)).flatten()

DM = DM.flatten()
HI = HI.flatten()


#Create Histogram

bins_DM = np.linspace(0,28,50)      #0 is the minimum value of the DM maps and 28 is the maximum value
bins_DM_mean = 0.5*(bins_DM[1:] + bins_DM[: -1] )     
Number_points = np.histogram(DM, bins=bins_DM)[0]
HI_total = np.histogram(DM, bins=bins_DM, weights=HI)[0]
HI_mean = HI_total / Number_points
HI_mean = np.nan_to_num(HI_mean)    #Change NaN values in array to 0

#Plot the points of DM vs HI and the best fit line using the mean values

plt.plot(bins_DM_mean, HI_mean, color = 'Red')   #Best Fit Line
plt.scatter(DM, HI, alpha = 0.5)                 #Scatter Plot
plt.show()

# Load DM Maps into one matrix
DM_map = np.zeros((1000, 1024))

for i in range(1000):
    DM_map[i] = (np.log(np.load('DM_new_map_%d.npy'%(i))+1)).flatten()
    
#Estimate HI using the best fit line and load into HI_estimation matrix
HI_estimation = np.zeros((1000, 1024))

for i in range(1000):
    for j in range(1024):
        HI_estimation[i,j] = np.interp(DM_map[i,j], bins_DM_mean, HI_mean)  #Interpolate the best fit line
        
np.save('HI_estimation.npy', HI_estimation)
        
