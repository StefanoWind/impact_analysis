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
import matplotlib.dates as mdates
import pandas as pd
import matplotlib
import xarray as xr
import glob
from scipy import stats
import matplotlib.gridspec as gridspec
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs

source=os.path.join(cd,'data/awaken/kp.turbine.z01.b0')
source_info=os.path.join(cd,'data/KP_info.xlsx')
turbine_id='H05'
vars_scada=['WROT.BlPthAngVal1_10m_Avg','WROT.BlPthAngVal2_10m_Avg','WROT.BlPthAngVal3_10m_Avg','WMET.HorWdSpd_10m_Avg','WTUR.W_10m_Avg']
sdate='2021-12-01T00:00:00'
edate='2023-03-04T00:00:00'
bins_ws=np.arange(0,25.1,0.5)
bins_pt=np.arange(-5,91)
bins_pw=np.arange(0,1.1,0.05)*2800
min_ws=0
max_ws=25

#graphics
labels={'WROT.BlPthAngVal1_10m_Avg':r'Pitch of blade 1 [$^\circ$]',
        'WROT.BlPthAngVal2_10m_Avg':r'Pitch of blade 2 [$^\circ$]',
        'WROT.BlPthAngVal3_10m_Avg':r'Pitch of blade 3 [$^\circ$]',
        'WMET.HorWdSpd_10m_Avg':r'Wind Speed [m s$^{-1}$]'}

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

ws_avg=(bins_ws[:-1]+bins_ws[1:])/2
pt_avg=(bins_pt[:-1]+bins_pt[1:])/2
pw_avg=(bins_pw[:-1]+bins_pw[1:])/2

#%% Main
for f in files[sel]:
    try:
        Data=xr.open_dataset(f)
    except:
        print(f'Could not open {f}')
        continue
    
    Data=Data.rename({'WTUR.DateTime':'time'})
    
    qc_flag=(Data['WMET.HorWdSpd_10m_Avg']>min_ws)*(Data['WMET.HorWdSpd_10m_Avg']<max_ws)
    Data=Data.where(qc_flag)
    
    datasets.append(Data[vars_scada])
   
    
    print(f'{f} done.')
    
SCD=xr.concat(datasets,dim='time')

N_pt={} 
for bid in range(1,4):
    N_pt[bid]=stats.binned_statistic_2d(SCD['WMET.HorWdSpd_10m_Avg'].values, 
                                     SCD[f'WROT.BlPthAngVal{bid}_10m_Avg'].values,
                                     SCD['WMET.HorWdSpd_10m_Avg'].values,
                                     statistic='count',
                                     bins=(bins_ws,bins_pt))[0]

N_pw=stats.binned_statistic_2d(SCD['WMET.HorWdSpd_10m_Avg'].values, 
                                 SCD['WTUR.W_10m_Avg'].values,
                                 SCD['WMET.HorWdSpd_10m_Avg'].values,
                                 statistic='count',
                                 bins=(bins_ws,bins_pw))[0]
        
#%% Plots
plt.figure(figsize=(18,8))
ctr=1
for v in vars_scada:
    ax=plt.subplot(len(vars_scada),1,ctr)
    plt.plot(SCD.time,SCD[v],'k')
    plt.ylim([np.nanpercentile(SCD[v],1),np.nanpercentile(SCD[v],99)])
    plt.xlabel('Time (UTC)')
    plt.ylabel(labels[v])
    plt.grid()
    ctr+=1
plt.tight_layout()


fig = plt.figure(figsize=(18,6))
gs = gridspec.GridSpec(1, 4, width_ratios=[1,1,1,0.05], wspace=0.2)

for bid in range(1,4):
    ax=fig.add_subplot(gs[0, bid-1])
    pc=plt.pcolor(ws_avg,pt_avg,np.log10(N_pt[bid].T/N_pt[bid].max()),cmap='inferno',vmin=-3,vmax=0)
    plt.xlabel(labels['WMET.HorWdSpd_10m_Avg'])
    if bid==1:
        plt.ylabel(r'Blade pitch [$^\circ$]')
    else:
        ax.set_yticklabels([])
    plt.title(f'Turbine {turbine_id}, Blade #{bid}')
    plt.grid()
cax = fig.add_subplot(gs[0, 3])
plt.colorbar(pc,cax=cax,label='Normalized count')
cax.set_yticks(np.arange(-3,1))
cax.set_yticklabels([r'$10^{'+str(i)+'}$' for i in np.arange(-3,1)])
plt.tight_layout()

fig = plt.figure()
pc=plt.pcolor(ws_avg,pw_avg,np.log10(N_pw.T/N_pw.max()),cmap='inferno',vmin=-3,vmax=0)
plt.xlabel(labels['WMET.HorWdSpd_10m_Avg'])
plt.ylabel(r'Active power [kW]')
ax.set_yticklabels([])
plt.title(f'Turbine {turbine_id}')
plt.grid()
plt.colorbar(pc,cax=cax,label='Normalized count')
cax.set_yticks(np.arange(-3,1))
cax.set_yticklabels([r'$10^{'+str(i)+'}$' for i in np.arange(-3,1)])
plt.tight_layout()