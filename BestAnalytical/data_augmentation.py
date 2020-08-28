import numpy as np
import matplotlib.pyplot as plt
root = '../maps'

# Function to create Rotations and Flip Configurations (8 total)

def rot_and_flip(input_data, maps, l, w):   

''' 
    input_data : name of the data (str)
    maps       : number of maps 
    l and w    : length and width of each map
'''
    #create containers for each configuration
    data_0   = np.zeros((maps, l, w), dtype = np.float32)
    data_90  = np.zeros((maps, l, w), dtype = np.float32)
    data_180 = np.zeros((maps, l, w), dtype = np.float32)
    data_270 = np.zeros((maps, l, w), dtype = np.float32)
    data_f1  = np.zeros((maps, l, w), dtype = np.float32)
    data_f2  = np.zeros((maps, l, w), dtype = np.float32)
    data_f3  = np.zeros((maps, l, w), dtype = np.float32)
    data_f4  = np.zeros((maps, l, w), dtype = np.float32)
    
    #create each configuration and load into respective containers
    for i in range(1000):
        data_0[i]   = np.load('%s/%s_%d.npy'%(root, input_data, i))
        data_90[i]  = np.rot90(data_0[i], -1)     # 90 degrees clockwise
        data_180[i] = np.rot90(data_90[i], -1)  # 180 degrees clockwise
        data_270[i] = np.rot90(data_180[i], -1) # 270 degrees clockwise 
        data_f1[i]  = np.flip(data_0[i], -1)      # flip original
        data_f2[i]  = np.flip(data_90[i], -1)   # flip 90 degrees clockwise
        data_f3[i]  = np.flip(data_180[i], -1)  # flip 180 degrees clockwise
        data_f4[i]  = np.flip(data_270[i], -1)  # flip 270 degrees clockwise

    # Create new container for all the configurations and load them in   
    dataset = np.zeros((maps*8, l, w), dtype = np.float32)
    for i in range(maps*8):
        dataset[i:1000] = data_0[i:1000]
        dataset[(i+1000):2000] = data_90[i:1000]
        dataset[(i+2000):3000] = data_180[i:1000]
        dataset[(i+3000):4000] = data_270[i:1000]
        dataset[(i+4000):5000] = data_f1[i:1000]
        dataset[(i+5000):6000] = data_f2[i:1000]
        dataset[(i+6000):7000] = data_f3[i:1000]
        dataset[(i+7000):8000] = data_f4[i:1000]

    return dataset
    #np.save('%s_data'%(input_data), dataset) #Save data
    
