# -*- coding: utf-8 -*-
"""
Read raw SCADA
"""
import os
cd=os.getcwd()
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import pandas as pd
import glob
from nptdms import TdmsFile
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs
channel='kp.turbine.z03.00'
source=glob.glob(os.path.join(cd,f'data/awaken/{channel}/*tdms'))[0]

#%% Main
if 'parquet' in source:
    Data = pd.read_parquet(source)
    _vars=np.unique(np.array(Data['tag']))
    for v in _vars:
        print(v)
elif 'tdms' in source:
    file = TdmsFile.read(source)
    for group in file.groups():
        print("Group:", group.name)
        for channel in group.channels():
            print(channel.name)
    
#%% Output
