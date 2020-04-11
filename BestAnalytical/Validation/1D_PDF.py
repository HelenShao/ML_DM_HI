import numpy as np 
import matplotlib.pyplot as plt

#Load the full true and predicted HI maps
HI_True = np.load('HI_Full_True.npy')
HI_Predicted = np.load('HI_Full_Predicted_2.npy')

#Print shapes 
print(np.shape(HI_True), np.shape(HI_Predicted))

#Plot images
plt.subplot(221)
plt.imshow(HI_True, vmin= 0, vmax = 12)
cbar = plt.colorbar()
plt.title('HI_True')

plt.subplot(222)
plt.imshow(HI_Predicted, vmin = 0, vmax = 12)
cbar = plt.colorbar()
plt.title('HI_Predicted')

plt.subplots_adjust(bottom=12, right=2, top=14)

#Flatten
HI_True_flat = HI_True.reshape(1,4194304)
HI_Predicted_flat = HI_Predicted.reshape(1, 4194304)

#Find min and max
minimum_p, maximum_p = np.min(HI_Predicted_flat), np.max(HI_Predicted_flat)
minimum_t, maximum_t = np.min(HI_True_flat), np.max(HI_True_flat)

#Create bins for predicted
bins_HI_p = np.linspace(-2,13,11)      #-2/13 are the min/max values of the HI Pred
Number_points_p = np.histogram(HI_Predicted_flat, bins=bins_HI_p)[0]
print(minimum_p, maximum_p, minimum_t, maximum_t)

#Create bins for true
bins_HI_t = np.linspace(-2, 13, 11)
Number_points_t = np.histogram(HI_True_flat, bins=bins_HI_t)[0]

#Plot distribution 
plt.plot(Number_points_p, label = 'HI_predicted')
plt.plot(Number_points_t, label = 'HI_True')
plt.yscale('log')
plt.ylabel('Probability Distribution Function')
plt.xlabel('Binned Values of HI Intensity')
plt.text(2, 10**3,'(Re-trained)')
plt.legend()
