# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:19:04 2019

@author: Vlasenko

Function for computing weights for each subdomain


"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import datetime as dt  # Python standard library datetime  module
import numpy as np
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
#from mpl_toolkits import addcyclic, shiftgrid
from sklearn.metrics.cluster import silhouette_score
import scipy.signal as scs
#from skimage.transform import rescale, resize 
import os
from scipy.stats.stats import pearsonr 

from ncl_extracter import ncdump
import xarray as xr

import pandas as pd 



wind = np.load('Wind_speed.npy', allow_pickle=True, fix_imports=True)
ival = np.load('Ival.npy', allow_pickle=True, fix_imports=True)
zen = np.load('Zenith.npy',allow_pickle=True, fix_imports=True)
shift = np.load('shift.npy',allow_pickle=True, fix_imports=True)





Nvar=[]
Pvar=[]
    



for ii in range(0,90000):
        
        a  = np.reshape(ival[2][ii+shift[2]],[17*11])
        d  = np.reshape(ival[5][ii+shift[5]],[17*11])
        
        a1  = wind[2][ii+shift[2]]
        d1  = wind[5][ii+shift[5]]
        
        if np.isnan(a)[1] == False and np.isnan(d)[1] == False:
            Nvar = Nvar+[(wind[0][ii],d1)]
            Pvar = Pvar+[(a,d)]




tr_y = np.asarray(Nvar)
tr_x = np.asarray(Pvar)


mx = np.mean(tr_x) 
tr_x = tr_x - mx 
ax = np.max(np.abs(tr_x))
tr_x = tr_x/ax
    
my = np.mean(tr_y) 
tr_y = tr_y - my 
ay = np.max(np.abs(tr_y))
tr_y = tr_y/ay

[l1,foo,bar] = tr_x.shape


l2 = int(3*l1/4)

test_x = tr_x[l2:,:,:]
train_x = tr_x[:l2,:,:]

test_y = tr_y[l2:,:]
train_y = tr_y[:l2,:]

yo=187
#%%
model = tf.keras.Sequential([
     layers.SimpleRNN(yo, activation='tanh',kernel_initializer='orthogonal', dropout=0.25),
     layers.Dense(yo, activation='tanh',kernel_initializer='orthogonal'),
     layers.Dense(2, activation='tanh',kernel_initializer='orthogonal')]) 

 # tf.optimizers.Adam()
model.compile(optimizer=tf.optimizers.Adam(0.0025251),
                   loss='mse',
                   metrics=['mae'])


print('start')
for i in range(0,150):
    model.fit(train_x,train_y, shuffle=True, epochs=1, batch_size=15
                   , verbose = 2)

    result = model.predict(test_x)
    Corr,Cp=pearsonr(test_y[:,0],result[:,0])
    print('Corr = ', Corr)

for i in range(0,150):
    model.fit(train_x,train_y, shuffle=True, epochs=1, batch_size=1500000
                   , verbose = 2)
    
    result = model.predict(test_x)
#     pathW = pathw+gas+prefix+str(num)
#     model.save_weights(pathW)
#     np.save(path+gas+prefix,result, allow_pickle=True, fix_imports=True )  



# #

