# -*- coding: utf-8 -*-
"""
Visualize loads long term trends
"""
import os
cd=os.getcwd()
import sys
import numpy as np
from matplotlib import pyplot as plt
import warnings
import pandas as pd
import matplotlib
import xarray as xr
import glob
from scipy import stats
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs

source=os.path.join(cd,'data/awaken/kp.turbine.z01.b0')
source_info=os.path.join(cd,'data/KP_info.xlsx')
sdate='2022-12-01T00:00:00'
edate='2023-03-04T00:00:00'
turbine_id='H05'
#
# var_x='WMET.HorWdSpd_10m_Avg'
var_x='WROT.BlPthSpt_10m_Avg'
vars_y=['WROT.BlPthAngVal1_10m_Avg','WROT.BlPthAngVal2_10m_Avg','WROT.BlPthAngVal3_10m_Avg']
        # 'WROT.BlPthSpt_10m_Avg','WTUR.W_10m_Avg']

bins={'WMET.HorWdSpd_10m_Avg':np.arange(0,25.1,0.5),
      'WROT.BlPthAngVal1_10m_Avg':np.arange(-5,91),
        'WROT.BlPthAngVal2_10m_Avg':np.arange(-5,91),
        'WROT.BlPthAngVal3_10m_Avg':np.arange(-5,91),
        'WROT.BlPthSpt_10m_Avg':np.arange(500,2000,10),
        'WTUR.W_10m_Avg':np.arange(0,1.1,0.05)*2800}
#qc
min_ws=0
max_ws=25

#graphics
labels={'WROT.BlPthAngVal1_10m_Avg':r'Pitch of blade 1 [$^\circ$]',
        'WROT.BlPthAngVal2_10m_Avg':r'Pitch of blade 2 [$^\circ$]',
        'WROT.BlPthAngVal3_10m_Avg':r'Pitch of blade 3 [$^\circ$]',
        'WROT.BlPthSpt_10m_Avg': 'Pitch setpoint',
        'WMET.HorWdSpd_10m_Avg':r'Wind Speed [m s$^{-1}$]',
        'WTUR.W_10m_Avg':'Active power [kW]'}

#%% Functions
def dates_from_files(files):
    import re
    '''
    Extract data from data filenames
    '''
    dates=np.array([],dtype='datetime64')
    for f in files:
        match = re.search( r"\b\d{8}\.\d{6}\b", os.path.basename(f))
        datestr=match.group()
        dates=np.append(dates,np.datetime64(f'{datestr[:4]}-{datestr[4:6]}-{datestr[6:8]}T{datestr[9:11]}:{datestr[11:13]}:{datestr[13:15]}'))
    
    return dates

def mid(x):
    return (x[1:]+x[:-1])/2

#%% Initialization

info=pd.read_excel(source_info).set_index('Turbine #')
turbine_flag=info['Wind Turbine Code'].loc[turbine_id].split('_')[-1].lower()
files=np.array(sorted(glob.glob(os.path.join(source,'*'+turbine_flag+'*'))))

dates=dates_from_files(files)
sel=(dates>np.datetime64(sdate))*(dates<np.datetime64(edate))

#zeroing
v=[]
time=np.array([],dtype='datetime64')
datasets = [] 

#%% Main

#concatenate data
for f in files[sel]:
    try:
        Data=xr.open_dataset(f)
    except:
        print(f'Could not open {f}')
        continue
    
    Data=Data.rename({'WTUR.DateTime':'time'})
    
    qc_flag=(Data['WMET.HorWdSpd_10m_Avg']>min_ws)*(Data['WMET.HorWdSpd_10m_Avg']<max_ws)
    Data=Data.where(qc_flag)
    
    datasets.append(Data[[var_x]+vars_y])
   
    print(f'{f} done.')
    
SCD=xr.concat(datasets,dim='time')

#caclulate curves
N={}
for v in vars_y:
    N[v]=stats.binned_statistic_2d(SCD[var_x].values, 
                                     SCD[v].values,
                                     SCD[v].values,
                                     statistic='count',
                                     bins=(bins[var_x],bins[v]))[0]

#%% Plots

for v in vars_y:
    fig = plt.figure()
    pc=plt.pcolor(mid(bins[var_x]),mid(bins[v]),np.log10(N[v].T/N[v].max()),cmap='inferno',vmin=-3,vmax=0)
    plt.xlabel(labels[var_x])
    plt.ylabel(labels[v])
    plt.title(f'Turbine {turbine_id}')
    plt.grid()
    cbar =plt.colorbar(pc,label='Normalized count')
    cbar.set_ticks(np.arange(-3,1))
    cbar.set_ticklabels([r'$10^{'+str(i)+'}$' for i in np.arange(-3,1)])
    plt.tight_layout()