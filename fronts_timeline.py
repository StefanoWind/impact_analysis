# -*- coding: utf-8 -*-
"""
Visualize occurence of weather fronts
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
import pandas as pd
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs
source=os.path.join(cd,'data','Periods_fronts_all.xlsx')



#%% Initialization
Data=pd.read_excel(source)
time1=[np.datetime64(t.replace(' ','T')) for t in Data['Start time']]

#%% Plots
plt.figure(figsize=(18,3))
plt.plot(time1,np.zeros(len(time1)),'-',color=(0,0,0.8),linewidth=3)
plt.plot(time1,np.zeros(len(time1)),'.',color=(0,0,0.8),markersize=10)
plt.bar(time1,Data['Intensity'],color=(0,0,0.8),width=np.timedelta64(3,'D'))
plt.xlim([np.datetime64('2023-01-01T00:00:00'),np.datetime64('2025-08-15T00:00:00')])
plt.ylabel('Front intensity')