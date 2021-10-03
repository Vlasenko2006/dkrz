# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:19:04 2019

@author: Vlasenko

Convolutional neural network for reconstructing missed wind speed from satelite measurements

"""

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


from scipy.stats.stats import pearsonr 
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
 


wind = np.load('Wind_speed.npy', allow_pickle=True, fix_imports=True) # put actual path to the data if needed

C = []    

shift = np.zeros([8,1])
Mval = np.zeros([8,1])

#  We find the maximal point where thw mesurements from the lost satelite, say s_id = 0 and other satelites
s_id = 0
for j in range(0,8): # iterating by satelits
    C = np.zeros([4000,1]) # 4000 must be measurements for one loop over the Earth
    for ii in range(0,4000):
        Corr,foo=pearsonr(wind[j][0:80000],wind[s_id][0+ii:80000+ii])
        C[ii] = Corr
        print(ii)
    shift[j] = np.argmax(C) # computing valocity shifts
    Mval[j] = np.max(C)




#%%
# We pack the data for the CNN. No lists, objects ect., only np.arrays. we want things to go fast

s = 12

Pvar=np.zeros([90000-int(max(shift)),7,2*s])
Nvar=np.zeros([90000-int(max(shift)),1])
c = 0
for ii in range(int(max(shift))+s,90000):   
        Nvar[c,0] = wind[0][ii]
        Pvar[c,0,:] = np.asarray(wind[1][ii-int(shift[1,0])-s:ii-int(shift[1,0])+s])
        Pvar[c,1,:] = np.asarray(wind[2][ii-int(shift[2,0])-s:ii-int(shift[2,0])+s])
        Pvar[c,2,:] = np.asarray(wind[3][ii-int(shift[3,0])-s:ii-int(shift[3,0])+s])
        Pvar[c,3,:] = np.asarray(wind[4][ii-int(shift[4,0])-s:ii-int(shift[4,0])+s]) 
        Pvar[c,4,:] = np.asarray(wind[5][ii-int(shift[5,0])-s:ii-int(shift[5,0])+s]) 
        Pvar[c,5,:] = np.asarray(wind[6][ii-int(shift[6,0])-s:ii-int(shift[6,0])+s]) 
        Pvar[c,6,:] = np.asarray(wind[7][ii-int(shift[7,0])-s:ii-int(shift[7,0])+s]) 
        c=c+1
        print(ii)

# Normalizing the data, to make estimates faster
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

[l1,foo, bar] = tr_x.shape


l2 = int(3*l1/4) # splitter for the training and test sets

test_x = tr_x[l2:,:,:]
train_x = tr_x[:l2,:,:]

test_y = np.reshape(tr_y[l2:],[len(tr_y[l2:]),1])
train_y = np.reshape(tr_y[:l2],[len(tr_y[:l2]),1])


#     The networks
model = tf.keras.Sequential()
model.add(Conv1D(2*s, 5, activation='relu', input_shape=(7,2*s)))
model.add(Flatten())
model.add(Dense(1, activation='tanh'))
model.summary()

model1 = tf.keras.Sequential()
model1.add(Conv1D(2*s, 3, activation='relu', input_shape=(7,2*s)))
model1.add(Flatten())
model1.add(Dense(1, activation='tanh'))
model1.summary()

model2 = tf.keras.Sequential()
model2.add(Conv1D(2*s, 3, activation='relu', input_shape=(7,2*s)))
model2.add(Flatten())
model2.add(Dense(1, activation='tanh'))
model2.summary()



print('start')
model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mae'])

model1.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mae'])

model2.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mae'])

for i in range (0,1):
    history = model.fit(train_x, train_y, epochs=1, 
                    validation_data=(test_x, test_y))
    history1 = model1.fit(train_x, train_y, epochs=1, 
                    validation_data=(test_x, test_y))
    history2 = model2.fit(train_x, train_y, epochs=1, 
                    validation_data=(test_x, test_y))
    
    result = 0.3333*(model.predict(test_x)+model1.predict(test_x)+model2.predict(test_x))
    Corr,Cp=pearsonr(test_y[:,0],result[:,0])
    print('corr = ',Corr)




# converting data back to its scales
result = result*ay+my
test_y = test_y*ay+my
