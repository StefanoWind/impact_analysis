# -*- coding: utf-8 -*-
"""
Explore data file
"""
import os
cd=os.getcwd()
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib.dates as mdates
import matplotlib
import xarray as xr
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs
channel='kp.turbine.z03.b0'
time_var={'kp.turbine.z01.b0':'WTUR.DateTime',
          'kp.turbine.z02.b0':'date',
          'kp.turbine.z03.b0':'time',
          'kp.turbine.z03.c0':'time'}
date_fmt={'kp.turbine.z01.b0':'%H:%M',
          'kp.turbine.z02.b0':'%H:%M',
          'kp.turbine.z03.b0':'%H:%M',
          'kp.turbine.z03.c0':'%H:%M'}

sub_var={'kp.turbine.z01.b0':'',
          'kp.turbine.z02.b0':'',
          'kp.turbine.z03.b0':'',
          'kp.turbine.z03.c0':['stat','mean']}



#%% Initialization
source=glob.glob(os.path.join(cd,f'data/awaken/{channel}/*nc'))[0]
Data=xr.open_dataset(source)
Data=Data.rename({time_var[channel]:'time'})

os.makedirs(os.path.join(cd,'figures',os.path.basename(source)[:-3]),exist_ok=True)

#%% Main
ctr=0
for v in Data.data_vars:
    plt.figure(figsize=(20,10))
    if sub_var[channel]!='':
        plt.plot(Data.time,Data[v].sel({sub_var[channel][0]:sub_var[channel][1]}),'.-k')
    else:
        plt.plot(Data.time,Data[v],'.-k')
    plt.xlabel('Time (UTC)')
    plt.ylabel(v)
    plt.title(f'Filename: {os.path.basename(source)}')
    plt.grid()
    if 'description' in Data[v].attrs:
        plt.ylabel(Data[v].attrs['description'])
    if 'units' in Data[v].attrs:
        plt.ylabel(plt.gca().get_ylabel()+' ['+Data[v].attrs['units']+']')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_fmt[channel]))
    plt.savefig(os.path.join(cd,'figures',os.path.basename(source)[:-3],f'{ctr:02d}.{v}.png'))
    plt.close()
    ctr+=1
    
    
