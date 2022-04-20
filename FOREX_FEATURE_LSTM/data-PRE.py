# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 09:34:11 2020

@author: yiming
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import yfinance as yf
from hmmlearn import hmm
from numpy import cumsum, log, polyfit, sqrt, std, subtract
import warnings

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action = 'ignore',category = SettingWithCopyWarning)





data = yf.download(" ^GSPC EURUSD=X USDCHF=X ", start="2019-07-01", end="2020-07-01")
# whatever financial time series we want to download, as well as the initial and end date
range_period = 180
best_lag = 27
# extract data and drop Null
data_EURUSD = (data['Adj Close']['EURUSD=X']).fillna(data['Adj Close']['EURUSD=X'].mean()).reset_index()
data_SP500 = (data['Adj Close']['^GSPC']/1000).fillna(data['Adj Close']['^GSPC'].mean()/1000).reset_index()

data_date_list = data.index.tolist()#date

df = pd.DataFrame( columns= ['Date','EURUSD=X','S&P','STD','Vol_global','Vol_local',\
                             'return','Price_MA','Primary risk',\
                             'FD','Energy','Correlation','STD_short','Price_LB',\
                             'Price_UB','Price_LB_Price_UB_Distance','Distance_MA'])
df['Date'] = data_date_list
df['EURUSD=X'] = data_EURUSD['EURUSD=X']
df['S&P'] = data_SP500['^GSPC']
#calculate the std within range 200
df['STD'] = df['EURUSD=X'].rolling(range_period).std(ddof=0)
df['return'] = (df['EURUSD=X'] - df['EURUSD=X'].shift(1)) / df['EURUSD=X'].shift(1)
df['return'] = df['return'].fillna(0)

#get the std_max to calculate the vol_global
df_std_max = df['STD'].max()
#calculate the vol_global
df['Vol_global'] = df['STD'] / df_std_max


#calculate the vol_local
df['Vol_local'] = df['STD'] / df['STD'].rolling(10).mean() / 3


#calculate the primary risk using np.ceil (round up)
df['Primary risk'] = (1/ df['Vol_global']).apply(np.ceil) / 10


#calculate the correlation
df['Correlation'] = 0.5*(df['EURUSD=X'].rolling(range_period).corr(df['S&P'])+1)


#calculate fd and hurst
FDs = list()
Hurst = list()
for i, data_row in df.iterrows():
    if i < len(df)+1-range_period:
        spy_close = df[i:i+range_period-1]
        spy_close = spy_close[['EURUSD=X']].copy()
        lag1, lag2 = 2, 20 # lag chosen 2 , 20
        lags = range(lag1, lag2)
            
        #Fractal Dimension calculation
        tau = [sqrt(std(subtract(spy_close[lag:], spy_close[:-lag]))) for lag in lags]
        m = polyfit(log(lags), log(tau), 1)
        hurst = m[0]*2
        fractal = 2-hurst[0]
        FDs.append(fractal)
        Hurst.append(hurst)
    else:
        break
df['FD'][range_period-1:len(df)+1] = FDs



from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
Vols = df['Vol_global'][range_period-1:len(df)+1]
FD = df['FD'][range_period-1:len(df)+1]
X = Vols.values.reshape(-1,1)
y = FD.values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Linear Regression model built on train data
regressor = LinearRegression() 

#training the algorithm
regressor.fit(X_train, y_train) 

#Calculating the coeff B of Fractal Dim
c = regressor.intercept_
B =1/c

#Calculating the coeff A of Vol
m = regressor.coef_
A = m.item()*B
#calculate the Energy
df['Energy'] = A*df['Vol_global'] + B*df['FD']
df['Price_MA'] = df['EURUSD=X'].rolling(best_lag).mean()
df['STD_short'] = df['EURUSD=X'].rolling(best_lag).std(ddof=0)
df['Price_LB'] = df['Price_MA'] - 2 * df['STD_short']
df['Price_UB'] = df['Price_MA'] + 2.02 * df['STD_short']
df['Price_LB_Price_UB_Distance'] = df['Price_UB']- df['Price_LB']
df['Distance_MA'] = df['Price_LB_Price_UB_Distance'].rolling(best_lag).mean()


"""
save as CSV
"""
df.to_csv("C:/Users/yiming/Feature Eng/EURUSD_1YEAR.csv")


######################################################################
######################################################################
######################################################################
range_period = 160
best_lag = 27
# extract data and drop Null
data_USDCHF = (data['Adj Close']['USDCHF=X']).fillna(data['Adj Close']['USDCHF=X'].mean()).reset_index()
data_SP500 = (data['Adj Close']['^GSPC']/1000).fillna(data['Adj Close']['^GSPC'].mean()/1000).reset_index()

data_date_list = data.index.tolist()#date

df = pd.DataFrame( columns= ['Date','USDCHF=X','S&P','STD','Vol_global','Vol_local',\
                             'return','Price_MA','Primary risk',\
                             'FD','Energy','Correlation','STD_short','Price_LB',\
                             'Price_UB','Price_LB_Price_UB_Distance','Distance_MA'])
df['Date'] = data_date_list
df['USDCHF=X'] = data_USDCHF['USDCHF=X']
df['S&P'] = data_SP500['^GSPC']
#calculate the std within range 200
df['STD'] = df['USDCHF=X'].rolling(range_period).std(ddof=0)
df['return'] = (df['USDCHF=X'] - df['USDCHF=X'].shift(1)) / df['USDCHF=X'].shift(1)
df['return'] = df['return'].fillna(0)

#get the std_max to calculate the vol_global
df_std_max = df['STD'].max()
#calculate the vol_global
df['Vol_global'] = df['STD'] / df_std_max


#calculate the vol_local
df['Vol_local'] = df['STD'] / df['STD'].rolling(10).mean() / 3


#calculate the primary risk using np.ceil (round up)
df['Primary risk'] = (1/ df['Vol_global']).apply(np.ceil) / 10


#calculate the correlation
df['Correlation'] = 0.5*(df['USDCHF=X'].rolling(range_period).corr(df['S&P'])+1)


#calculate fd and hurst
FDs = list()
Hurst = list()
for i, data_row in df.iterrows():
    if i < len(df)+1-range_period:
        spy_close = df[i:i+range_period-1]
        spy_close = spy_close[['USDCHF=X']].copy()
        lag1, lag2 = 2, 20 # lag chosen 2 , 20
        lags = range(lag1, lag2)
            
        #Fractal Dimension calculation
        tau = [sqrt(std(subtract(spy_close[lag:], spy_close[:-lag]))) for lag in lags]
        m = polyfit(log(lags), log(tau), 1)
        hurst = m[0]*2
        fractal = 2-hurst[0]
        FDs.append(fractal)
        Hurst.append(hurst)
    else:
        break
df['FD'][range_period-1:len(df)+1] = FDs



from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
Vols = df['Vol_global'][range_period-1:len(df)+1]
FD = df['FD'][range_period-1:len(df)+1]
X = Vols.values.reshape(-1,1)
y = FD.values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Linear Regression model built on train data
regressor = LinearRegression() 

#training the algorithm
regressor.fit(X_train, y_train) 

#Calculating the coeff B of Fractal Dim
c = regressor.intercept_
B =1/c

#Calculating the coeff A of Vol
m = regressor.coef_
A = m.item()*B
#calculate the Energy
df['Energy'] = A*df['Vol_global'] + B*df['FD']
df['Price_MA'] = df['USDCHF=X'].rolling(best_lag).mean()
df['STD_short'] = df['USDCHF=X'].rolling(best_lag).std(ddof=0)
df['Price_LB'] = df['Price_MA'] - 2 * df['STD_short']
df['Price_UB'] = df['Price_MA'] + 2.02 * df['STD_short']
df['Price_LB_Price_UB_Distance'] = df['Price_UB']- df['Price_LB']
df['Distance_MA'] = df['Price_LB_Price_UB_Distance'].rolling(best_lag).mean()


"""
save as CSV
"""
df.to_csv("C:/Users/yiming/Feature Eng/USDCHF_1YEAR.csv")