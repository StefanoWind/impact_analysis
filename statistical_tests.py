# -*- coding: utf-8 -*-
"""
Check pdf of faulty and healthy turbine
"""
import os
cd=os.getcwd()
import numpy as np
from matplotlib import pyplot as plt
import warnings
import pandas as pd
import matplotlib
from scipy.stats import mannwhitneyu
import utils
import xarray as xr
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

period=30#[days] days prior to failure
min_ws=10#[m/s] min wind speed
max_ws=25#[m/s] max wind speed
min_power=2800*0.01#[kW] min power
max_power=2800*1.02#[kW] max power
pvalue=0.05#pvalue of Wilconcox test

#graphics 
id_plot='B08x'

#%% Initialization
failure=pd.read_excel(source_failure)
info=pd.read_excel(source_info).set_index('Turbine #')

#zeroing
Data={}
Data_all={}
Data_healthy={}
Data_faulty={}

os.makedirs(os.path.join(cd,'figures',id_plot+'_sample'),exist_ok=True)

#%% Main
for failed,shutdown in zip(failure['Substation Name - Turbine'],failure['Shutdown Date']):
    turbine_id=failed.split('-')[1]
    turbine_flag=info['Wind Turbine Code'].loc[turbine_id].split('_')[-1].lower()
    date_fail=np.datetime64(str(shutdown).replace(' ','T'))
    
    #concatenate data from different SCADA channels
    datasets=[]
    for s in sources:
        file=glob.glob(os.path.join(s,f'*{turbine_flag}.nc'))
        if len(file)==1:
            datasets.append(xr.open_dataset(file[0]))
        
    if len(datasets)>0:
        Data[turbine_id]=xr.concat(datasets, dim='time')
        
        #global qc
        qc_ws=(Data[turbine_id].ws_avg>=min_ws)*(Data[turbine_id].ws_avg<=max_ws)
        qc_power=(Data[turbine_id].power_avg>=min_power)*(Data[turbine_id].power_avg<=max_power)
        Data[turbine_id]=Data[turbine_id].where(qc_ws*qc_power)
        print(f'Removed {np.sum(qc_ws.values==0)} off-range wind speed points in {turbine_id}')
        print(f'Removed {np.sum(qc_power.values==0)} off-range power points in {turbine_id}')
        
        #add ptp
        _vars=np.unique([v[:-4] for v in  Data[turbine_id].data_vars])
        for v in _vars:
            if f'{v}_min' in Data[turbine_id].data_vars and f'{v}_max' in Data[turbine_id].data_vars:
                Data[turbine_id][f'{v}_ptp']=Data[turbine_id][f'{v}_max']-Data[turbine_id][f'{v}_min']
        #specific qc
        for v in Data[turbine_id].data_vars:
            if 'status' not in v:
                dv=np.gradient(Data[turbine_id][v])
                qc_dv=np.abs(dv)!=0
                Data[turbine_id][v]=Data[turbine_id][v].where(qc_dv)
                print(f'Removed {np.sum(qc_dv==0)} flat points in {v} of {turbine_id}')
        
        #flag and combine data
        healthy=Data[turbine_id].time<date_fail-np.timedelta64(period,'D')
        faulty=(Data[turbine_id].time>=date_fail-np.timedelta64(period,'D'))*(Data[turbine_id].time<=date_fail)
        Data[turbine_id]['failed']=faulty
        ctr=0
        for v in Data[turbine_id].data_vars:
            x=Data[turbine_id][v]
            if v in Data_healthy.keys():
                Data_all[v]=np.append(Data_all[v],x)
            else:
                Data_all[v]=x
            x=Data[turbine_id][v].where(healthy)
            if v in Data_healthy.keys():
                Data_healthy[v]=np.append(Data_healthy[v],x[~np.isnan(x)])
            else:
                Data_healthy[v]=x[~np.isnan(x)]
            x=Data[turbine_id][v].where(faulty)
            if v in Data_faulty.keys():
                Data_faulty[v]=np.append(Data_faulty[v],x[~np.isnan(x)])
            else:
                Data_faulty[v]=x[~np.isnan(x)]
            
            #plot sample time series
            if turbine_id==id_plot:
                plt.figure(figsize=(18,8))
                plt.plot(Data[turbine_id].time,Data[turbine_id][v],'.-k',markersize=1)
                plt.plot([date_fail,date_fail],
                         [plt.gca().get_ylim()[0],plt.gca().get_ylim()[1]],'-r',linewidth=2)
                plt.plot([date_fail-np.timedelta64(period,'D'),date_fail-np.timedelta64(period,'D')],
                          [plt.gca().get_ylim()[0],plt.gca().get_ylim()[1]],'-r',linewidth=2)
                plt.ylabel(v)
                plt.xlabel('Time UTC')
                plt.grid()
                plt.savefig(os.path.join(cd,'figures',turbine_id+'_sample',f'{ctr:02.0f}.{v}.png'))
                plt.close()
            ctr+=1
            
            
#RF
X=np.zeros((len(Data_all['power_avg']),len(Data_all.keys())))
i=0
vars_plot=[]
for v in Data_all.keys():
    if 'status' not in v and 'failed' not in v and 'amb_temperature_avg' not in v:
        vars_plot.append(v)
        X[:,i]=Data_all[v]
        i+=1
X=X[:,:i]
y=Data_all['failed'] 

importance,importance_std,y_pred,test_mae,train_mae,best_params,y_test,predicted_test=utils.RF_feature_selector(X[:,:-5],y)          
            
#%% Plots

#test on the variable
fig=plt.figure(figsize=(18,10))
cols=int(np.ceil(len(Data_healthy.keys())/4))
ctr=1
for v in Data_healthy.keys():
    try:
        stat, p = mannwhitneyu(Data_healthy[v], Data_faulty[v], alternative="two-sided")
        
        ax=plt.subplot(4,cols,ctr)
        plt.hist(Data_healthy[v],density=True,bins=100,color='g',alpha=0.25,label=f'More than {period} days before failure')
        plt.hist(Data_faulty[v],density=True,bins=100,color='r',alpha=0.25,label=f'{period} days before failure')
        plt.plot([np.nanmedian(Data_healthy[v]),np.nanmedian(Data_healthy[v])],[ax.get_ylim()[0],ax.get_ylim()[1]],'--g')
        plt.plot([np.nanmedian(Data_faulty[v]),np.nanmedian(Data_faulty[v])],[ax.get_ylim()[0],ax.get_ylim()[1]],'--r')
        plt.xlabel(v)
        if p>pvalue:
            plt.text(ax.get_xlim()[0],ax.get_ylim()[1]*0.9,f'Wilcoxon: {p:.2f}')
        else:
            plt.text(ax.get_xlim()[0],ax.get_ylim()[1]*0.9,f'Wilcoxon: {p:.2f}',fontweight='bold')
        if (ctr-1)/cols==int((ctr-1)/cols):
            plt.ylabel('Probability')
        ax.set_yticklabels([])
        
        
        plt.grid()
        ctr+=1
    except:
        pass
plt.tight_layout()

fig=plt.figure(figsize=(18,10))
cols=int(np.ceil(len(Data_healthy.keys())/4))
ctr=1
for v in Data_healthy.keys():
    try:
        stat, p = mannwhitneyu(np.gradient(Data_healthy[v]), np.gradient(Data_faulty[v]), alternative="two-sided")
        
        ax=plt.subplot(4,cols,ctr)
        plt.hist(np.gradient(Data_healthy[v]),density=True,bins=100,color='g',alpha=0.25,label=f'More than {period} days before failure')
        plt.hist(np.gradient(Data_faulty[v]),density=True,bins=100,color='r',alpha=0.25,label=f'{period} days before failure')
        plt.plot([np.nanmedian(np.gradient(Data_healthy[v])),np.nanmedian(np.gradient(Data_healthy[v]))],
                 [ax.get_ylim()[0],ax.get_ylim()[1]],'--g')
        plt.plot([np.nanmedian(np.gradient(Data_faulty[v])),np.nanmedian(np.gradient(Data_faulty[v]))],
                 [ax.get_ylim()[0],ax.get_ylim()[1]],'--r')
        plt.xlabel(r'$\Delta$'+v)
        if p>pvalue:
            plt.text(ax.get_xlim()[0],ax.get_ylim()[1]*0.9,f'Wilcoxon: {p:.2f}')
        else:
            plt.text(ax.get_xlim()[0],ax.get_ylim()[1]*0.9,f'Wilcoxon: {p:.2f}',fontweight='bold')

        if (ctr-1)/cols==int((ctr-1)/cols):
            plt.ylabel('Probability')
        ax.set_yticklabels([])
        
        plt.grid()
        ctr+=1
    except:
        pass
        
plt.tight_layout()

plt.figure()
plt.bar(vars_plot,importance,color='Gray')
plt.errorbar(np.arange(len(vars_plot)),importance,importance_std, color='k',linestyle='none',capsize=5)
plt.xticks(rotation=90)
plt.tight_layout()