import numpy as np
import sys,os
import Pk_library as PKL

BoxSize = 75.0 #Mpc/h
MAS     = 'CIC'
threads = 1
f1 = 'HI_Predicted_Full2.npy'
f2 = 'HI_Full_True.npy'

HI_pred = np.load(f1);  #HI_pred = HI_pred/np.mean(HI_pred) - 1.0
HI_true = np.load(f2);  #HI_true = HI_true/np.mean(HI_true) - 1.0
print HI_pred.shape
print HI_true.shape
print np.min(HI_true),np.max(HI_true),np.mean(HI_true)
print np.min(HI_pred),np.max(HI_pred),np.mean(HI_pred)


Pk = PKL.Pk_plane(HI_true, BoxSize, MAS, threads)
np.savetxt('Pk_true.txt', np.transpose([Pk.k, Pk.Pk]))

Pk = PKL.Pk_plane(HI_pred, BoxSize, MAS, threads)
np.savetxt('Pk_pred.txt', np.transpose([Pk.k, Pk.Pk]))
