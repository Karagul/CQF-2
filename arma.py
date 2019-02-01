# -*- coding: utf-8 -*-

from __future__ import division
import pandas as pd
import mysql.connector
from pandas.io import sql
from sqlalchemy import create_engine
import pymysql
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import scipy.optimize as so
from scipy.optimize import brentq
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from arch import arch_model
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import statsmodels.tsa.stattools as st
from statsmodels.tsa.stattools import adfuller, acf, pacf



df1=pd.read_csv('spy.csv')

df1=pd.read_sql(sql,yconnect_cd)

df=df1[['Date','Adj Close']]
df['price']=df['Adj Close']
del df['Adj Close']

df=df.ix[245:,:]
df['last_price']=df['price'].shift()


df['real_r']=(df['price']-df['last_price'])/df['last_price']
df['r']=(df['price']-df['last_price'])/df['last_price']




df=df.dropna()
df=df.reset_index(drop=True)
#***********calculate annualized std**********
std_l=[]
for i in range(4,len(df)):
 
    std=np.std(df['r'][i-4:i])
    std_l=np.append(std_l,std)


dfa=df.ix[4:,:]
dfa['std']=std_l
#**************************


dfa['std_last']=dfa['std'].shift()
dfa=dfa.dropna()
dfa=dfa.reset_index(drop=True)

def garch(a,b,c,std_last,rc_last):
    garch_vol=a+b*std_last**2+c*rc_last**2
    return garch_vol


mean_p=np.mean(dfa['price'])
std_p=np.std(dfa['price'])
dfa['p_normalize']=(dfa['price']-mean_p)/std_p


mean_vol=np.mean(dfa['std'])
std_vol=np.std(dfa['std'])
dfa['std_normalize']=(dfa['std']-mean_vol)/std_vol

ts = pd.Series(np.random.randn(500), index=pd.date_range('2010-01-01', periods=500))
dfa1=dfa[:500].copy()

#################


all_num=0
predict_list=[]
vol_list=[]
except_num=0
for i in range(len(dfa1)-5):
    try:
        pq=st.arma_order_select_ic(list(dfa1['r'][i:i+5]),max_ar=3,max_ma=3,ic=['bic']).bic_min_order
        arma=ARMA(list(dfa1['r'][i:i+5]),pq).fit(disp=False)
        output1=arma.forecast()
        garch = arch_model(dfa1['r'][i:i+5],vol='Garch', p=1, o=0, q=1, dist='Normal').fit()
        output2=garch.forecast()

        vol=np.sqrt(output2.variance.tail(1).values)
        vol_list=np.append(vol_list,vol)
        
        if vol>=dfa1['std'][i+4]:
            output=output1[0]-2*vol
        else:
            output=output1[0]+8*vol
        
        predict_list=np.append(predict_list,output)
    except:
        garch = arch_model(dfa1['r'][i:i+5],vol='Garch', p=1, o=0, q=1, dist='Normal').fit()
        output2=garch.forecast()
        vol=np.sqrt(output2.variance.tail(1).values)
        output=np.mean(list(dfa1['r'][i:i+5]))
        predict_list=np.append(predict_list,output)
        vol_list=np.append(vol_list,vol)
        except_num+=1


dfa_com=dfa1[5:500]
dfa_com=dfa_com[['Date','r','real_r']]
#dfa_com['predict']=predict_list

dfa_com['predict']=predict_list
dfa_com['p_vol']=vol_list

dfa_com=dfa_com.reset_index(drop=True)

dfa_com['diff']=(dfa_com['r']-dfa_com['predict'])/dfa_com['p_vol']
dfa_com['check']=0
for j in range(len(dfa_com)):
    if dfa_com['r'][j]*dfa_com['predict'][j]>0:
        dfa_com['check'][j]=1

print dfa_com['check'].sum()



dfa_com['signal']=0
for s in range(len(dfa_com)):
    if dfa_com['predict'][s]>0:
        dfa_com['signal'][s]=1
    else:
        dfa_com['signal'][s]=-1



dfa_com['real_return']=dfa_com['signal']*dfa_com['real_r']+1

dfa_com['cumr']=dfa_com['real_return'].cumprod()
print dfa_com['real_return'].prod()-1

print (dfa_com['real_r']+1).prod()-1


plt.plot(dfa_com['Date'],dfa_com['real_return'].cumprod(),'r')
plt.plot(dfa_com['Date'],(dfa_com['real_r']+1).cumprod(),'b')
plt.legend(['our strategy','buy and hold'])
plt.show()








