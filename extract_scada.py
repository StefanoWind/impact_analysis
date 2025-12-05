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

source_failure=os.path.join(cd,'data','King Plains Pitch Bearing Crack List_2025_11_07.xlsx')
channels=['awaken/kp.turbine.z01.b0','awaken/kp.turbine.z02.c0']
source_info=os.path.join(cd,'data/KP_info.xlsx')
sdate='2023-01-01 00:00:00'

_vars={}
_vars['awaken/kp.turbine.z01.b0']=  {'power':'WTUR.W',
                                    'pitch1':'WROT.BlPthAngVal1',
                                    'pitch2':'WROT.BlPthAngVal2',
                                    'pitch3':'WROT.BlPthAngVal3',
                                    'amb_temperature':'WMET.EnvTmp',
                                    'wd':'WMET.HorWdDir',
                                    'ws':'WMET.HorWdSpd',
                                    'yaw':'WNAC.Dir',
                                    'status':'WTUR.TurSt'}
_vars['awaken/kp.turbine.z02.c0']= {'power':'WTUR.W',
                                    'pitch1':'WROT.BlPthAngVal1',
                                    'pitch2':'WROT.BlPthAngVal2',
                                    'pitch3':'WROT.BlPthAngVal3',
                                    'amb_temperature':'WNAC.EnvTmp',
                                    'wd':'WMET.HorWdDir',
                                    'ws':'WMET.HorWdSpd',
                                    'yaw':'WNAC.Dir',
                                    'status':'WTUR.TurSt'}

stats=['Avg','Min','Max']

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
with open(os.path.join(cd,'configs/config.yaml'), 'r') as fid:
    config = yaml.safe_load(fid)
    
failure=pd.read_excel(source_failure).iloc[:2,:]
info=pd.read_excel(source_info).set_index('Turbine #')

a2e = DAP('wdh.energy.gov',confirm_downloads=False)
a2e.setup_two_factor_auth(username=config['username'], password=config['password'])

#%% Main
for failed,shutdown in zip(failure['Substation Name - Turbine'],failure['Shutdown Date']):
    turbine_id=failed.split('-')[1]
    turbine_flag=info['Wind Turbine Code'].loc[turbine_id].split('_')[-1].lower()
    
    edate=str(shutdown)
    
    for channel in channels:
    
        #download
        
        _filter = {
            'Dataset': channel,
            'date_time': {
                'between': [sdate.replace('-','').replace(':','').replace(' ',''),
                            edate.replace('-','').replace(':','').replace(' ','')]
            },
            'file_type':'nc',
            'ext1': turbine_flag
        }
        
        os.makedirs(os.path.join(cd,'data',channel),exist_ok=True)
        # a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel),replace=False)
        
        #stack
        datasets=[]
        files=np.array(sorted(glob.glob(os.path.join(cd,'data',channel,f'*{turbine_flag}.nc'))))
        dates=dates_from_files(files)
        sel=(dates>=np.datetime64(sdate.replace(' ','T')))*(dates<=np.datetime64(edate.replace(' ','T')))
        if np.sum(sel)>0:
            for f in files[sel]:
                try:
                    Data=xr.open_dataset(f)
                except:
                    print(f'Could not open {f}')
                    continue
                
                Data=Data.rename({'WTUR.DateTime':'time'})
                
                Data_sel=xr.Dataset()
                for v in _vars[channel]:
                    for s in stats:
                        if f'{_vars[channel][v]}_10m_{s}' in Data.data_vars:
                            Data_sel[f'{v}_{s.lower()}']=Data[f'{_vars[channel][v]}_10m_{s}']
                          
                datasets.append(Data_sel)
                print(f'{f} appended')
                
            #save
            os.makedirs(os.path.join(cd,'data',channel[:-2]+'d0'),exist_ok=True)
            Output=xr.concat(datasets,dim='time')
            Output.to_netcdf(os.path.join(cd,'data',channel[:-2]+'d0',f'{channel.split("/")[1][:-2]+"d0"}.{turbine_flag}.nc'))

    
    
