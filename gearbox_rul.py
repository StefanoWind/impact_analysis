# -*- coding: utf-8 -*-
"""
PINN for gearbox bearing
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib as mpl
from datetime import timedelta
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.size'] = 12
mpl.rcParams['savefig.dpi']=300

plt.close('all')

#%% Inputs
source=os.path.join(cd,'data','HighSpeedShaftBearing.csv')

start=300# start row
t_offset=1475# t offset
lr = 0.0001 # learning rate
ur = 0.5 # percentage of available data
lam_eq = 1 # physics weight
lam_data = 1  # data weight
m_c = 100;
noise_std = 0.2
t_0 = 0.05 # initial value of HI
t_f = 2 #final value of HI
N = 1000 # numer of collocation points of the NN
m = 10 # number of neurons
y1_0 = 1475

IterMax = 100
IterTol = 1e-16

type_act = 2 # activation functions
LB = -1 # Lower boundary for weight and bias samplings
UB = -LB # Upper boundary for weight and bias samplings

a_0 = 0.05
a_f = 1

#%% Functions
def act(x,w,b):
    act = (np.exp(b + w*x) - np.exp(- b - w*x))/(np.exp(b + w*x) + np.exp(- b - w*x));
    actd =(w*np.exp(b + w*x) + w*np.exp(- b - w*x))/(np.exp(b + w*x) + np.exp(- b - w*x)) - ((np.exp(b + w*x) - np.exp(- b - w*x))*(w*np.exp(b + w*x) - w*np.exp(- b - w*x)))/(np.exp(b + w*x) + np.exp(- b - w*x))**2;
    actdd = (2*(np.exp(b + w*x) - np.exp(- b - w*x))*(w*np.exp(b + w*x) - w*np.exp(- b - w*x))**2)/(np.exp(b + w*x) + np.exp(- b - w*x))**3 - (2*(w*np.exp(b + w*x) + w*np.exp(- b - w*x))*(w*np.exp(b + w*x) - w*np.exp(- b - w*x)))/(np.exp(b + w*x) + np.exp(- b - w*x))**2 - (w**2*np.exp(- b - w*x) - w**2*np.exp(b + w*x))/(np.exp(b + w*x) + np.exp(- b - w*x)) - ((np.exp(b + w*x) - np.exp(- b - w*x))*(w**2*np.exp(- b - w*x) + w**2*np.exp(b + w*x)))/(np.exp(b + w*x) + np.exp(- b - w*x))**2;
    actddd = (w**3*np.exp(- b - w*x) + w**3*np.exp(b + w*x))/(np.exp(b + w*x) + np.exp(- b - w*x)) - (6*(np.exp(b + w*x) - np.exp(- b - w*x))*(w*np.exp(b + w*x) - w*np.exp(- b - w*x))**3)/(np.exp(b + w*x) + np.exp(- b - w*x))**4 + (6*(w*np.exp(b + w*x) + w*np.exp(- b - w*x))*(w*np.exp(b + w*x) - w*np.exp(- b - w*x))**2)/(np.exp(b + w*x) + np.exp(- b - w*x))**3 + ((np.exp(b + w*x) - np.exp(- b - w*x))*(w**3*np.exp(- b - w*x) - w**3*np.exp(b + w*x)))/(np.exp(b + w*x) + np.exp(- b - w*x))**2 - (3*(w**2*np.exp(- b - w*x) + w**2*np.exp(b + w*x))*(w*np.exp(b + w*x) + w*np.exp(- b - w*x)))/(np.exp(b + w*x) + np.exp(- b - w*x))**2 + (3*(w**2*np.exp(- b - w*x) - w**2*np.exp(b + w*x))*(w*np.exp(b + w*x) - w*np.exp(- b - w*x)))/(np.exp(b + w*x) + np.exp(- b - w*x))**2 + (6*(np.exp(b + w*x) - np.exp(- b - w*x))*(w**2*np.exp(- b - w*x) + w**2*np.exp(b + w*x))*(w*np.exp(b + w*x) - w*np.exp(- b - w*x)))/(np.exp(b + w*x) + np.exp(- b - w*x))**3 ;
    
    return act,actd,actdd,actddd

#%% Initialization
Data=pd.read_csv(source).iloc[start:]

HI=Data[' HI Value'].values
dt=pd.to_datetime(Data["Timestamp"], format="%Y%m%dT%H%M%SZ").values
tnum=dt.astype("int64")/10**9 
t=(tnum-tnum[0])/3600-t_offset
n=len(HI)

#%% Main

#sort data
sort=np.argsort(HI)

HI_sort=HI[sort]
t_sort=t[sort]

HI_uniform=np.linspace(HI_sort[0],HI_sort[-1],n)
t_uniform=np.interp(HI_uniform,HI_sort,t_sort)

#MC
HI_pert_mat=np.outer(HI,np.ones((1,m_c)))*(1+np.random.randn(n,m_c)*noise_std)

HI_uniform_mat=np.zeros((n,m_c))
t_uniform_mat= np.zeros((n,m_c))
for i in range(m_c): 
    sort=np.argsort(HI_pert_mat[:,i])
    HI_sort=HI_pert_mat[sort,i]
    t_sort=t[sort]
    HI_uniform_mat[:,i]=np.linspace(HI_sort[0],HI_sort[-1],n)
    t_uniform_mat[:,i]=np.interp(HI_uniform_mat[:,i],HI_sort,t_sort)
    
    
HI_cut_mat = HI_uniform_mat[int(lr*n):int(ur*n),:]
t_HI_cut_mat = t_uniform_mat[int(lr*n):int(ur*n),:]

x =np. linspace(-1,1,N)# Discretization of collocation points

x_data = 2*(HI_cut_mat - np.min(HI_cut_mat))/(np.max(HI_cut_mat) - np.min(HI_cut_mat)) - 1 # Discretization of collocation points
N_x_data = len(HI_cut_mat)


t_step = (t_f-t_0)/1
t_tot = np.arange(t_0,t_f,t_step)
n_t = len(t_tot)
t_domain = np.linspace(t_0,t_f,N)


#training
Z = np.zeros((N,m))
z = np.zeros((N,1))

sol1 = np.zeros((N,m_c))
sol1_dot = np.zeros((N,m_c))

sol1_data = np.zeros((N_x_data,m_c))

training_err_vec = np.zeros((n_t-1,m_c))


for k in range(m_c):
    a_0 = HI_pert_mat[0,k]
    weight = np.random.uniform(LB,UB,m)
    bias = np.random.uniform(LB,UB,m)
    
    h= np.zeros((N,m))
    hd= np.zeros((N,m))
    hdd= np.zeros((N,m))
    
    for i in range(N):
        for j in range(m):
            h[i, j], hd[i, j], hdd[i,j],_ = act(x[i],weight[j], bias[j])
            
    h0 = h[0,:]
            
    h_data= np.zeros((N_x_data,m))
    hd_data= np.zeros((N_x_data,m))
    hdd_data= np.zeros((N_x_data,m))
    
    for i in range(N_x_data):
        for j in range(m):
            h_data[i, j], hd_data[i, j], hdd_data[i,j],_ = act(x_data[i,k],weight[j], bias[j])
            
    h0_data = h_data[0,:]
    
    
    for i in range(n_t-1):
    
        xi = np.ones((m,1))
     
        c_i = (x[-1] - x[0]) / (t_tot[i+1] - t_tot[i])
        
        y1 = (h-h0)*xi + y1_0
        y1_dot = c_i*hd*xi   
        y1_D = (h_data-h0_data)*xi + y1_0




