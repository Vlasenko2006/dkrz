# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 20:47:36 2018

@author: anrey

This file onlyb prepares the data for the CNN

"""

'''

        http://dx.doi.org/10.1175/1520-0477(1996)077<0437:TNYRP>2.0.CO;2
'''
import datetime as dt  # Python standard library datetime  module
import numpy as np
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import matplotlib.pyplot as plt

from sklearn.metrics.cluster import silhouette_score
import scipy.signal as scs
import os
from scipy.stats.stats import pearsonr 
from ncl_extracter import ncdump
import xarray as xr
import pandas as pd 





file_name ='cygnss_four_channels.nc' # or put your path to the dat
path='DKRZ/'



#--  data file name
fname  = path+file_name



ds_disk = xr.open_dataset(fname)

ival = ds_disk.brcs.values
ws = ds_disk.windspeed.values



# copieng values from ds_disk, 

wind_speed = []
time =       []
delay =      []
mmap =       []          
zenith =     []
l      =     []
ival   =     []
zenith    =  []

Wind_speed = []
Zenith     = []
Ival       = []
   
for i in range(1,9):                                          # unpacking the data
    s1 = ds_disk.where(ds_disk.spacecraft_num ==i, drop=True) 
    wind_speed = wind_speed + [(s1.windspeed.values)]
    time = time + [(s1.ddm_timestamp_utc.values)]
    zenith  = zenith + [(s1.zenith_code_phase.values)]
    ival = ival + [(ds_disk.brcs.values)]
    
    
    l = l+[(len(time[i-1]))]

lm = np.min(np.asarray(l))
larg = np.argmin(np.asarray(l))
ref = 1000000173.0
D = np.zeros([8,lm])
#%%
for s_id in range(0,8):          # synchronizing the daya 
    tmp1 = []
    tmp2 = []
    tmp3 = []
    tmp4 = []
    for i in range(0,lm):
        d = (time[s_id][i] - time[larg][i])/np.timedelta64(1, 'ns')        
              
        if d > -ref*1.09 and d < ref*1.09:            # all satelites ge the signal at the same time
            tmp1 = tmp1 + [(wind_speed[s_id][i])]
            tmp2 = tmp2 + [(zenith[s_id][i])]
            tmp3 = tmp3 + [(ival[s_id][i])]

            
        if d < -ref*1.09:      # looking for the missed signals forwards in time
            inc = 1
            d1=d
            while d1< -ref*1.09:
                
                d1 = (time[s_id][i+inc] - time[larg][i])/np.timedelta64(1, 'ns')
                inc+=1
                if inc>850:
                    d = 0
                if (inc+i)>(lm-1):
                    d1 = 0
                    i = lm
                    
            print('i = ', i, 'inc = ', inc )
            if (inc+i)<(lm-1):
                tmp1 = tmp1 + [(wind_speed[s_id][i+inc])]
                tmp2 = tmp2 + [(zenith[s_id][i])]
                tmp3 = tmp3 + [(ival[s_id][i])]                
                
                

######            
        if d > ref*1.09:   # looking for the missed signals backwards in tiume
            inc = 1
            d1=d
            while d1 > ref*1.09:
                
                d1 = (time[s_id][i-inc] - time[larg][i])/np.timedelta64(1, 'ns')
                inc+=1
                if inc>300:
                    d1 = 0
                if (i-inc)<1:
                    d1 = 0
                    i = lm
                
            print('i = ', i, 'inc = - ', inc )        
            if (i-inc)>0:
                tmp1 = tmp1 + [(wind_speed[s_id][i-inc])]
                tmp2 = tmp2 + [(zenith[s_id][i])]
                tmp3 = tmp3 + [(ival[s_id][i])]
######                      
            

    
    Wind_speed = Wind_speed+[(np.asarray(tmp1))]
    Zenith     = Zenith+[(np.asarray(tmp2))]
    Ival       = Ival + [(tmp3)]
        

       
        

D[D>10e12]=0
D = D/ref    

for i in range(0,6):
    wind_speed[i] = np.reshape(np.asarray(Wind_speed[i][:]),[len((Wind_speed[i][:]))])



np.save('Wind_speed',Wind_speed, allow_pickle=True, fix_imports=True)
np.save('Ival',Ival, allow_pickle=True, fix_imports=True)
np.save('Zenith',Zenith, allow_pickle=True, fix_imports=True)













