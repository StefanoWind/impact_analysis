# -*- coding: utf-8 -*-
"""
Visualize data availability on WDH
"""

import os
cd=os.getcwd()
import numpy as np
from matplotlib import pyplot as plt
from doe_dap_dl import DAP
import warnings
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs
source='data/WHOI_WFIP3_barge_bowstern_GPS_22-Oct-2024.dat'
headers='yyyy mm dd HH MM SS bow_lon bow_lat stern_lon stern_lat'
username='sletizia'
password='pass_DAP1506@'
channels=['awaken/kp.turbine.z01.b0','awaken/kp.turbine.z02.b0','awaken/kp.turbine.z02.c0','awaken/kp.turbine.z03.b0','awaken/kp.turbine.z03.c0']
file_types={'awaken/kp.turbine.z01.b0':'nc','awaken/kp.turbine.z02.b0':'nc','awaken/kp.turbine.z03.b0':'nc','awaken/kp.turbine.z03.c0':'nc'}
ext1s={'awaken/kp.turbine.z01.b0':'wt001','awaken/kp.turbine.z02.b0':'wt001','awaken/kp.turbine.z03.b0':'h5','awaken/kp.turbine.z03.c0':'h5'}
sdate='20200101000000'#start date for data search
edate='20251031000000'#end date for data search


#%% Functions
def datenum(string,format="%Y-%m-%d %H:%M:%S.%f"):
    '''
    Turns string date into unix timestamp
    '''
    from datetime import datetime
    num=(datetime.strptime(string, format)-datetime(1970, 1, 1)).total_seconds()
    return num

def datestr(num,format="%Y-%m-%d %H:%M:%S.%f"):
    '''
    Turns Unix timestamp into string
    '''
    from datetime import datetime
    string=datetime.utcfromtimestamp(num).strftime(format)
    return string

def dap_search(channel,sdate,edate,file_type='',ext1='',time_search=30):
    '''
    Wrapper for a2e.search to avoid timeout:
        Inputs: channel name, start date, end date, file format, extention in WDH name, number of days scanned at each loop
        Outputs: list of files mathing the criteria
    '''
    dates_num=np.arange(datenum(sdate,'%Y%m%d%H%M%S'),datenum(edate,'%Y%m%d%H%M%S'),time_search*24*3600)
    dates=[datestr(d,'%Y%m%d%H%M%S') for d in dates_num]+[edate]
    search_all=[]
    for d1,d2 in zip(dates[:-1],dates[1:]):
        _filter = {
            'Dataset': channel,
            'date_time': {
                'between': [d1,d2]
            },
        }
        if file_type!='':
            _filter['file_type']=file_type
        if ext1!='':
            _filter['ext1']=ext1

        search=a2e.search(_filter)
        
        if search is None:
            print('Invalid authentication')
            return None
        else:
            search_all+=search
        print(channel+'-'+d1)
    
    return search_all

#%% Initalization
a2e = DAP('wdh.energy.gov',confirm_downloads=False)
a2e.setup_two_factor_auth(username=username, password=password)
    
#%% Main

files={}
time_file={}
for c in channels:
    files[c]=dap_search(c, sdate,edate,file_type=file_types[c],ext1=ext1s[c])
    print(f'{len(files[c])} found in {c}')

    time_file[c]=np.array([np.datetime64(f'{f["data_date"][:4]}-{f["data_date"][4:6]}-{f["data_date"][6:8]}T00:00:00') for f in files[c]])


#%% Plots
plt.figure(figsize=(16,4))

ctr=0
for c in channels:
    plt.plot(time_file[c],np.zeros(len(time_file[c]))-ctr,'.',markersize=10,label=f'{c.split("/")[1]}*{ext1s[c]}*{file_types[c]}')
    ctr+=1
plt.yticks([])
plt.grid()
plt.legend(draggable=True)