from __future__ import division
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import csv
from arch import arch_model
from sklearn.svm import SVR
from scipy import stats






#ticker='SPY'
ticker = 'DDM'




yconnect_cd = create_engine('mysql+pymysql://root:root@localhost:3306/db_one?charset=utf8')


sql="select * from raw_stockdata."+ticker
df1=pd.read_sql(sql,yconnect_cd)

'''
#ticker='AMLP US Equity'
ticker = 'SPY'

df1 = pd.read_excel('/Users/samlee/Documents/smu/term four/quant trading/project/code/SPY_qt.xls')

#df1 = pd.read_csv('/Users/samlee/Documents/smu/term four/quant trading/project/data/19etf/' + ticker + '.csv')

#####################################
df1['Date'] = pd.to_datetime(df1['Date'])

'''



df = df1.copy()

df['price'] = df['Close']
df=df.ix[245:,:]

df['last_price']=df['price'].shift()


df['real_r']=(df['price']-df['last_price'])/df['last_price']
df['spot_return'] = df['real_r']



df=df.dropna()
df=df.reset_index(drop=True)



#***********calculate annualized std**********

#####################################
#dfa = df.copy()
dfa=df.ix[4:,:]
dfa=dfa.reset_index(drop = True)


dfa['std'] = 0
for i in range(len(dfa)):
    dfa['std'].iloc[i] = 0.5 * (np.log(dfa['High'].iloc[i]) - np.log(dfa['Low'].iloc[i]))**2 - 0.386 * (np.log(dfa['Close'].iloc[i]) - np.log(dfa['Open'].iloc[i]))**2
#############################


end = 3500
dfa=dfa[:end]

dfa.dropna()
dfa = dfa.reset_index(drop = True)



dfa['price'] = np.log(dfa['price'])



###########normalized data################





#**************svm part****************
#r_prd = []

diff = 5



day = np.arange(1,diff +1,1)
day_oj = day.reshape((diff,1))


bbc = 2
'''
bbc is 0 when bull market, and 4 when bear bear market 
'''
signal = []
for i in range(len(dfa) - diff):

    svr_md = SVR(kernel = 'rbf', gamma = 0.1)

    std_arr = np.array(dfa['std'][i:i +diff]).reshape(diff,1)
    day_std = np.concatenate((day_oj, std_arr), axis = 1)

    y = np.array(dfa['price'][i:i+diff]).reshape((diff,1))
    svr_md.fit(day_std, y)



    garch = arch_model(dfa['real_r'][i:i+diff],vol='Garch', p=1, o=0, q=1, dist='Normal').fit()
    output2=garch.forecast()
    vol=np.sqrt(output2.variance.tail(1).values)
    ret = svr_md.predict([[diff + 1, vol]])[0]



    if ret >= dfa['price'].iloc[i+diff-1]:
        signal = np.append(signal, 1)
  
    elif ret < dfa['price'].iloc[i+diff - 1] and vol *(1-0.1*(5-1*bbc)) > sorted(dfa['std'][i:i+diff ], reverse = True)[bbc]:
        signal = np.append(signal, -1*np.clip((bbc*(1+bbc/10)),1,5)/5)

    else:
        signal = np.append(signal, 2)
        #signal =np.append(signal, 1)
    

    if ret > dfa['price'].iloc[i+diff]:
        bbc = np.clip(bbc+2, 0, 4)
    elif ret < dfa['price'].iloc[i+diff]:
        bbc = np.clip(bbc-2, 0, 4) 
    
  

dfa_com=dfa[diff:end]
dfa_com['predict'] = signal


dfa_com = dfa_com.reset_index(drop = True)

for s in range(1, len(dfa_com)):
    if dfa_com['predict'].iloc[i] == 2:
        dfa_com['predict'].iloc[i] = dfa_com['predict'].iloc[i - 1]
        

dfa_com['real_return'] = dfa_com['predict'] * dfa_com['spot_return'] 
dfa_com['cum_r'] = (dfa_com['real_return']+1).cumprod()


#    return dfa_com

#dfa_com = strategy4one(df1)

###########################t-test###################


zero = [0.025/250] * len(dfa_com)

t_test_ret = stats.ttest_ind(zero, dfa_com['real_return'], axis = 0, equal_var = False)
print 'p_value of null hyp risk-free rate > our return : ', t_test_ret.pvalue/2
print 't_stat of null hyp risk-free rate > our return : ', t_test_ret.statistic 
################################################


print 'our strategy net value on '+ticker+' : ', (dfa_com['real_return']+1).prod()
print 'buy and hold net value on '+ticker+' : ', (dfa_com['spot_return']+1).prod()

dfa_com.index = dfa_com['Date']


(dfa_com['real_return']+1).cumprod().plot(color = 'r')
#plt.show()
(dfa_com['spot_return']+1).cumprod().plot(color = 'b')

#plt.plot(dfa_com['Date'], (dfa_com['real_return']+1).cumprod(), 'r')
#plt.plot(dfa_com['Date'], (dfa_com['spot_return']+1).cumprod(),'b')
plt.legend(['our strategy','buy and hold'])
plt.title(ticker+' compund cumulative return')
plt.show()



(dfa_com['real_return']).cumsum().plot(color = 'r')
(dfa_com['spot_return']).cumsum().plot(color = 'b')

#plt.plot(dfa_com['Date'], dfa_com['real_return'].cumsum(), 'r')
#plt.plot(dfa_com['Date'],dfa_com['spot_return'].cumsum() ,'b')
plt.legend(['out strategy','buy and hold'])
plt.title(ticker+' simple cumulative return')
plt.show()

print 'the annualized return of our strategy : ',dfa_com['real_return'].sum()/(len(dfa_com)/252)
print 'the annualized return of buy and hold : ',dfa_com['spot_return'].sum()/(len(dfa_com)/252)



#check the probability when our return is lower than spot return
check = 0
for i in range(len(dfa_com)):
    if dfa_com['real_return'][i] < dfa_com['spot_return'][i]:
        check+=1




print 'our failure rate :',check/len(dfa_com)




#######################################

def Sum_profit(day_profit):
    sum_profit = day_profit.cumsum()
    return sum_profit
    

def Annual_profit(trade_date, sum_profit):

    data = {'trade_date': trade_date,
            'sum_profit': sum_profit}
    
    dataframe = pd.DataFrame(data)
    trade_days = len(dataframe.index)
    annual_profit = dataframe.sum_profit.iloc[-1]*252/trade_days
    return annual_profit, trade_days

def Max_drawdown(trade_date, sum_profit):
    
    data = {'trade_date': trade_date,
            'sum_profit': sum_profit}
    dataframe = pd.DataFrame(data)
    dataframe['max2here'] = pd.expanding_max(dataframe['sum_profit'])
    dataframe['drawdown'] = dataframe['sum_profit'] - dataframe['max2here']
    temp = dataframe.sort_values(by = 'drawdown').iloc[0]
    max_drawdown = temp.drawdown
    max_drawdown_enddate = temp.trade_date.strftime('%Y-%m-%d')
    sub_dataframe = dataframe[dataframe.trade_date <= max_drawdown_enddate]
    max_drawdown_startdate = sub_dataframe.sort_values(by = 'sum_profit', ascending = False).iloc[0]['trade_date'].strftime('%Y-%m-%d')

    return max_drawdown, max_drawdown_startdate, max_drawdown_enddate



def Day_win_chance(trade_date, day_profit):
    data = {'trade_date': trade_date,
            'day_profit': day_profit}
    dataframe = pd.DataFrame(data)
    day_win_chance = len(dataframe[dataframe['day_profit'] > 0 ])/len(dataframe)
    return day_win_chance

def Max_sequent_days(trade_date, day_profit):

    data = {'trade_date': trade_date,
            'day_profit': day_profit}
    dataframe = pd.DataFrame(data)
    if dataframe.day_profit[0] > 0:
        dataframe.loc[0, 'count'] = 1
    else:
        dataframe.loc[0, 'count'] = -1
    for i in dataframe.index[1:]:
        if dataframe.day_profit[i] > 0 and dataframe.day_profit[i - 1] > 0:
            dataframe.loc[i, 'count'] = dataframe.loc[i-1,'count'] + 1
        elif dataframe.day_profit[i] <= 0 and dataframe.day_profit[i - 1] <= 0:
            dataframe.loc[i, 'count'] = dataframe.loc[i-1,'count']-1
        elif dataframe.day_profit[i] > 0 and dataframe.day_profit[i - 1] <= 0:
            dataframe.loc[i, 'count'] = 1
        elif dataframe.day_profit[i] <= 0 and dataframe.day_profit[i - 1] > 0:
            dataframe.loc[i, 'count'] = -1

    
    dataframe.count = list(dataframe['count'])
    return max(dataframe.count), min(dataframe.count)


def VIX(day_profit):
    return np.std(day_profit)*np.sqrt(252)


def Sharp_ratio(annual_profit, VIX):
    return (annual_profit - 0.025)/VIX


def Infromation_ratio(day_profit, benchmark_profit):
    diff = pd.Series(day_profit - benchmark_profit)
    return diff.mean() * 252/(diff.std() * np.sqrt(252))


def Draw_Line_chart(trade_date, sum_profit, sum_benchmark_profit):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(trade_date, sum_profit, color = 'r', label = 'sum_profit')
    ax.set_title('The line chart of sum_profit')
    ax.set_xlabel('trade_date')
    ax.set_ylabel('sum_profit')
    ax.legend(loc = 'best')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(trade_date, sum_profit, color = 'r', label = 'sum_profit')
    ax.plot(trade_date, sum_benchmark_profit, color = 'b', label ='sum_benchmark_profit')
    ax.set_title('Comaprison diagram')
    ax.set_xlabel('trade_date')
    ax.set_ylabel('sum_profit')
    ax.legend(loc = 'best')
    plt.show()


dfa_com = dfa_com.reset_index(drop = True)

trade_date = dfa_com['Date']
day_profit = dfa_com['real_return']
benchmark_profit = dfa_com['spot_return']

sum_profit = Sum_profit(day_profit)
sum_benchmark_profit = Sum_profit(benchmark_profit)
VIX = VIX(day_profit)
annual_profit = Annual_profit(trade_date, sum_profit)[0]

print 'the annualized return of our strategy in '+ticker +' is ', annual_profit
print 'the max_drawdown of our strategy in '+ticker+' is ', Max_drawdown(trade_date, sum_profit)
print 'the day_win_chance of our strategy in '+ticker+' is ', Day_win_chance(trade_date, day_profit)
print 'the Max_sequent_days of our strategy in '+ticker+' is ', Max_sequent_days(trade_date, day_profit)
print 'the Sharp_ratio of our strategy in '+ticker+' is ', Sharp_ratio(annual_profit, VIX)
print 'the Infromation_ratio of our strategy in '+ticker+' is ', Infromation_ratio(day_profit, benchmark_profit)



##### RISK -- VaR ####

def val_at_risk(array):
    array = sorted(array)
    return array[(int)(0.05*len(array))]

df_var = dfa_com[['Date','Close']]
df_var['Date'] = pd.to_datetime(dfa_com['Date'])
yr1 = df_var['Date'].dt.year
#mn1 = df_var['Date'].dt.month
startyear = yr1[0]
endyear = yr1[len(yr1)-1]
#month = mn1[0]
#df_var['Month']=(df_var['Date'].dt.year-year)*12 + df_var['Date'].dt.month ## creating month indices
df_var['Year'] = yr1
df_var['PnL'] = df_var['Close'].diff(1) ## daily PnL based on close price
df_var = df_var.dropna()
years = np.arange(startyear,endyear,1)
temp = []
h_var = []

for i in range(1,df_var.shape[0]-1):
    if (df_var['Year'][i]==df_var['Year'][i+1]):
        temp.append(df_var['PnL'][i])
    else:
        temp.append(df_var['PnL'][i])
        h_var.append(val_at_risk(temp))
        temp = []
    
plt.plot(years,h_var)
plt.title('Value at Risk - '+ticker)
plt.ylabel('VaR')
plt.show()        
    















