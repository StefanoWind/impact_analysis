# -*- coding: utf-8 -*-
"""
Check pdf of faulty and healthy turbine
"""
import os
cd=os.getcwd()
import sys
import numpy as np
from matplotlib import pyplot as plt
import warnings
from doe_dap_dl import DAP
import pandas as pd
import matplotlib
import xarray as xr
import yaml
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs


sources=[os.path.join(cd,'data','awaken/kp.turbine.z01.d0'),
         os.path.join(cd,'data','awaken/kp.turbine.z02.d0')]
source_info=os.path.join(cd,'data/KP_info.xlsx')

source_failure=os.path.join(cd,'data','King Plains Pitch Bearing Crack List_2025_11_07.xlsx')

period=30#[days]

#%% Initialization
failure=pd.read_excel(source_failure).iloc[:2,:]
info=pd.read_excel(source_info).set_index('Turbine #')

Data={}
Data_healthy={}
Data_faulty={}

#%% Main
for failed in failure['Substation Name - Turbine']:
    turbine_id=failed.split('-')[1]
    turbine_flag=info['Wind Turbine Code'].loc[turbine_id].split('_')[-1].lower()
    
    datasets=[]
    for s in sources:
        file=glob.glob(os.path.join(s,f'*{turbine_flag}.nc'))
        if len(file)==1:
            datasets.append(xr.open_dataset(file[0]))
    if len(datasets)>0:
        Data[turbine_id]=xr.concat(datasets, dim='time')
        
    healthy=Data[turbine_id].time<Data[turbine_id].time[-1]-np.timedelta64(period,'D')
    faulty=Data[turbine_id].time>=Data[turbine_id].time[-1]-np.timedelta64(period,'D')
    for v in Data[turbine_id].data_vars:
        if v in Data_healthy.keys():
            Data_healthy[v]=np.append(Data_healthy[v],Data[turbine_id][v].where(healthy,drop=True))
        else:
            Data_healthy[v]=Data[turbine_id][v].where(healthy,drop=True)
        if v in Data_faulty.keys():
            Data_faulty[v]=np.append(Data_faulty[v],Data[turbine_id][v].where(faulty,drop=True))
        else:
            Data_faulty[v]=Data[turbine_id][v].where(faulty,drop=True)
            
for v in Data_healthy.keys():
    try:
        plt.figure()
        plt.hist(Data_healthy[v],color='g',alpha=0.25)
        plt.hist(Data_faulty[v],color='r',alpha=0.25)
        plt.xlabel(v)
    except:
        pass
        
    