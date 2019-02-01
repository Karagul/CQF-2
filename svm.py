from __future__ import division
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import csv
from arch import arch_model
from sklearn.svm import SVR



ticker='SPY'


#ticker='IBM'pictu

yconnect_cd = create_engine('mysql+pymysql://root:root@localhost:3306/db_one?charset=utf8')


sql="select * from raw_stockdata."+ticker
df1=pd.read_sql(sql,yconnect_cd)


df = df1.copy()
#df=df1[['Date','Adj Close']]
df['price']=df['Adj Close']
del df['Adj Close']

df=df.ix[245:,:]


df=df.ix[245:,:]
df['last_price']=df['price'].shift()


df['real_r']=(df['price']-df['last_price'])/df['last_price']
df['spot_return'] = df['real_r']



df=df.dropna()
df=df.reset_index(drop=True)



#***********calculate annualized std**********
std_l=[]
for i in range(4,len(df)):
 
    std=np.std(df['real_r'][i-4:i])
    std_l=np.append(std_l,std)


dfa=df.ix[4:,:]
dfa['std']=std_l
dfa=dfa.reset_index(drop = True)

dfa=dfa[:500]


dfa['r'] = 0
for i in range(len(dfa)):
    if dfa['real_r'][i] >= 0:
        dfa['r'][i] = 1
    else:
        dfa['r'][i] = -1

dfa['next_r']=dfa['r'].shift(-1)
dfa.dropna()
dfa = dfa.reset_index(drop = True)





del dfa['Dividends']
del dfa['Stock Splits']






###########normalized data################

def normal(col_name):
    mean = np.mean(col_name)
    std = np.std(col_name)
    col_name = (col_name - mean)/std
    return col_name

#**************svm part****************
r_prd = []

diff = 10

col_name = ['Open','High','Low','price','Volume','real_r']
for name in col_name:
    dfa[name] = normal(dfa[name])
    #print dfa[name]





for i in range(len(dfa) - diff):

    for name in col_name:
        dfa[name][i:i+diff] = normal(dfa[name][i:i+diff])
    svr_md = SVC(kernel = 'rbf', gamma = 0.1)
    svr_md.fit(dfa[['Open','High','Low','price','Volume','real_r']][i:i+diff],dfa['r'][i:i+diff])

    for name in col_name:
        dfa[name][i:i+diff+1] = normal(dfa[name][i:i+diff+1])
    ret = svr_md.predict(dfa[['Open','High','Low','price','Volume','real_r']].iloc[i+diff])[0]

    r_prd = np.append(r_prd, ret)
    


print len(r_prd)
#print r_prd

r_prd = r_prd[:len(r_prd) -1 ]




dfa_com=dfa[11:500]
dfa_com['predict'] = r_prd

dfa_com = dfa_com.reset_index(drop = True)


dfa_com['real_return'] = dfa_com['predict'] * dfa_com['spot_return'] +1
dfa_com['cum_r'] = dfa_com['real_return'].cumprod()

print dfa_com['real_return'].prod()-1
print (dfa_com['spot_return']+1).prod()-1


plt.plot(dfa_com['Date'], dfa_com['real_return'].cumprod(), 'r')
plt.plot(dfa_com['Date'], (dfa_com['spot_return']+1).cumprod(),'b')
plt.legend(['out strategy','buy and hold'])
plt.show()

print dfa_com


#######################
