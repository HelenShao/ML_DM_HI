import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
DATA_DM = np.load('DM_maps_new.npy')
DATA_HI = np.load('HI_maps_new.npy')
DATA_ESTI = np.load('HI_estimated_large.npy')

# Function to create Rotations and Flip Configurations (8)

def rot_flip_func(input_data, maps, l, w):
    
    ''' 
        input_data : name of the data (DM_maps_new.npy, HI_maps_new.npy) (str)
        maps       : number of maps in input_data
        l and w    : length and width of each map
    '''
    data_0   = input_data
    
    #create containers to load the maps in
    data_90  = np.zeros((maps, l, w), dtype = np.float32)
    data_180 = np.zeros((maps, l, w), dtype = np.float32)
    data_270 = np.zeros((maps, l, w), dtype = np.float32)
    data_f1  = np.zeros((maps, l, w), dtype = np.float32)
    data_f2  = np.zeros((maps, l, w), dtype = np.float32)
    data_f3  = np.zeros((maps, l, w), dtype = np.float32)
    data_f4  = np.zeros((maps, l, w), dtype = np.float32)

    #create each configuration and load into respective containers
    for i in range(maps):
        data_90[i]  = np.rot90(data_0[i], -1)     # 90 degrees clockwise
        data_180[i] = np.rot90(data_90[i], -1)    # 180 degrees clockwise
        data_270[i] = np.rot90(data_180[i], -1)   # 270 degrees clockwise 
        data_f1[i]  = np.flip(data_0[i], -1)      # flip original
        data_f2[i]  = np.flip(data_90[i], -1)     # flip 90 degrees clockwise
        data_f3[i]  = np.flip(data_180[i], -1)    # flip 180 degrees clockwise
        data_f4[i]  = np.flip(data_270[i], -1)    # flip 270 degrees clockwise

    # Create new container for all the configurations and load them in   
    dataset_new = np.zeros((maps*8, l, w), dtype = np.float32)
   
    for i in range(maps):
        dataset_new[i:maps] = data_0[i]
        dataset_new[(i+maps):maps*2] = data_90[i]
        dataset_new[(i+maps*2):maps*3] = data_180[i]
        dataset_new[(i+maps*3):maps*4] = data_270[i]
        dataset_new[(i+maps*4):maps*5] = data_f1[i]
        dataset_new[(i+maps*5):maps*6] = data_f2[i]
        dataset_new[(i+maps*6):maps*7] = data_f3[i]
        dataset_new[(i+maps*7):maps*8] = data_f4[i]

    return dataset_new
  
# Create new transformed datasets (large)
DM = rot_flip_func(DATA_DM, 98304, 32, 32)
HI = rot_flip_func(DATA_HI, 98304, 32, 32)
ESTI = rot_flip_func(DATA_ESTI, 98304, 32, 32)

#Print shape of the final data_large.npy files
print(np.shape(ESTI))

#SAVE DATASETS
np.save('DM_transform_large', DM)
np.save('HI_transform_large', HI)
np.save('Esti_transform_large', ESTI)
