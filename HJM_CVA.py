# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:47:00 2018

@author: DELL
"""
import copy as copylib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#######data input and simulations#######
dat =  pd.read_csv('hjm_parameters.csv')
del dat['Tenor']
curve_spot = np.array(dat.iloc[4,:]).T
mc_vols = np.array(dat.iloc[1:4,:])
mc_drift = np.array(dat.iloc[0,:]).T     
proj_timeline = np.linspace(0,5,501)
mc_tenors = np.linspace(0,25,51)
                      
def simulation(f, tenors, drift, vols, timeline):
    assert type(tenors)==np.ndarray
    assert type(f) == np.ndarray
    assert type(drift)==np.ndarray
    assert type(timeline)==np.ndarray
    assert len(f)==len(tenors)
    len_tenors = len(tenors)
    len_vols = len(vols)
    yield timeline[0], copylib.copy(f)
    for it in range(1, len(timeline)):
        t = timeline[it]
        dt = t - timeline[it-1]
        sqrt_dt = np.sqrt(dt)
        fprev = f
        f = copylib.copy(f)
        random_numbers = [np.random.normal() for i in range(len_vols)]
        for iT in range(len_tenors):
            val = fprev[iT] + drift[iT] * dt
            sum = 0
            for iVol, vol in enumerate(vols):
                sum += vol[iT] * random_numbers[iVol]
            val += sum * sqrt_dt
            iT1 = iT+1 if iT<len_tenors-1 else iT-1   
            dfdT = (fprev[iT1] - fprev[iT]) / (iT1 - iT)
            val += dfdT * dt
            f[iT] = val
        yield t,f

proj_rates = []
for i, (t, f) in enumerate(simulation(curve_spot, mc_tenors, mc_drift, mc_vols, proj_timeline)):
    proj_rates.append(f)
proj_rates = pd.DataFrame(proj_rates)


simu_fwd = proj_rates.iloc[[0,99,499],:]
proj_fwd = proj_rates.iloc[:,[0,2,10,25]]
hang_biao = pd.DataFrame(np.linspace(0,5,501))

plt.plot(mc_tenors, simu_fwd.transpose()), plt.xlabel('Tenor Time'), plt.ylabel('forward rates'),\
plt.title('Simulated Forward Curves'), plt.legend(['1y','5y','10y']),plt.figure(figsize=(6, 4)),plt.show()

mc_tenor_1 = pd.DataFrame(mc_tenors).iloc[0:11]
plt.plot(hang_biao, proj_fwd), plt.xlabel('Future Historic Time'), plt.ylabel('Forward Rates'),\
plt.title('Projection of Forward Rate(at selected fixed tenors)'),plt.legend(['today','1y','5y','10y']) ,plt.show() 

###########turn fwd rates into libor rates for the 6M IRS############
libor_rates = 1/0.5 * (np.exp(proj_rates* 0.5)-1)
libor_rates = libor_rates.iloc[:,0:11]
libor_rates.columns= ['0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0','4.5',
                  '5.0']
libor_rates = pd.concat([hang_biao,libor_rates],axis=1)
libor_rates = libor_rates.set_index([0])
K = float(libor_rates.iloc[0,0]) #set a constant K 

libor_rates = libor_rates.iloc[[0,50,100,150,200,250,300,350,400,450,500],:]        
                 
ZCB = 1.0 / (1 + 0.5 * libor_rates)
ZCB.iloc[0, :] = 1.0
ZCB = ZCB.cumprod() 
ZCB_mean = pd.Series(index=ZCB.index, data=np.mean(ZCB, axis=1))
DF = pd.DataFrame(index=ZCB.index, columns=list(ZCB.index))
DF.iloc[0, :] = ZCB_mean

for index, row in DF.iterrows():
    if index == 0.0:
        continue
    x = DF.loc[0.0][row.index]/DF.loc[0.0, index]
    x[x > 1] = 0 
    DF.loc[index, :] = x

label = ['0.0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0', '4.0-4.5',
                  '4.5-5.0']

DF = DF.iloc[0,:]
DF = (DF+ DF.shift(-1))/2.0
DF = DF.iloc[:-1]
DF.index = label
DF = pd.DataFrame(DF)

MtM = (libor_rates-K)
EE = np.maximum(MtM,0)

PFE  = pd.Series( data=np.percentile(EE, q=97.5, axis=1))
PFE = (PFE+ PFE.shift(-1))/2.0
PFE = PFE.iloc[:-1]
PFE.index = label
PFE = pd.DataFrame(PFE)     
     
PFE.plot(subplots=False, kind='line',  color = 'orange',legend=False),plt.xlabel('Tenor Period'),plt.ylabel('PFE'),\
plt.title('97.5% PFE for Different Tenor Period'),plt.show() 


EE = pd.DataFrame(np.median(EE,axis = 1))
EE = (EE+ EE.shift(-1))/2.0
EE = EE.iloc[:-1]
EE.index = label

EE.plot(subplots=False, kind='bar', edgecolor= 'Black',width=1.0,legend=False),plt.xlabel('Tenor Period'),plt.ylabel('EE'),\
plt.title('Expected Exposure for Different Tenor Period'),plt.show() 


PD = pd.read_excel("D:/CQF/PD.xlsx",sheetname = 'PD')
PD = PD.iloc[:,1]
PD.index = label
PD = pd.DataFrame(PD)

PD.plot(subplots=False, kind='bar', color = 'yellow',edgecolor= 'Black',width=1.0,legend=False),plt.xlabel('Tenor Period'),plt.ylabel('PD'),\
plt.title('Default Probability for Different Tenor Period'),plt.show() 

RR = 0.4 * np.ones(10)
LGD = pd.DataFrame((1-RR) * np.ones(10))
LGD.index = label

CVA = LGD.iloc[:,0] * EE.iloc[:,0] * DF.iloc[:,0] * PD.iloc[:,0]

CVA.plot(subplots=False, kind='bar', color = 'pink',edgecolor= 'Black',width=1.0,legend=False),plt.xlabel('Tenor Period'),plt.ylabel('CVA'),\
plt.title('CVA for Different Tenor Period'),plt.show() 

CVA = CVA.sum() #0.00012488384046064848 FOR PORSCHE AG


