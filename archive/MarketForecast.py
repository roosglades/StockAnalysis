#%% Libraries
import os
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta

import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBClassifier
from xgboost import plot_importance

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from functions import *

import warnings
warnings.filterwarnings("ignore")


#%% Todo

# // TODO: get ARIMA prediction for 5 day doesn't do shit

# TODO: LSTM for 5 day prediction
# TODO: set ARIMA as an input using only 1 day forecsast

#%% User Input
Stock           = 'AAPL' #ticker
Predict         = 'close_return_log'
Lookback        = 2.5 # years

H               = 5 #forecast horizon in days
AH              = 1 #forecast for ARIMA
ARIMA_PreTrain  = 1 # pretrain ARIMA in years 
ARIMA_Predict   = 'Close'

# Plot Folder
Plots = 'D:/StockAnalytics/test'

# model parms
ScaleAll    = True
test_split  = 0.10 # train/test split
BatchSize   = 8 # number of samples to update weights
#BatchSizes  = list(range(8, 32, 8))
TimeStep    = 20 # 2 months used in LSTM model
#TimeSteps  = list(range(10, 90, 10))
Epoch       = 100 # number of times to run through the data
#Epochs     = list(range(50,550,50))
Node        = 128 # number of LSTM Node
#Nodes      = list(range(16,144,16))

#predict current/past
current = False

# if current=False
selected_date = '2020-01-01'

#%% Pre
if not os.path.exists(Plots):
    os.makedirs(Plots)

#%% Grabbing Data
print('Grabbing Stock Market Data...')
today_raw       = datetime.today()
today           = today_raw.strftime("%Y-%m-%d")

if current:
    date_select = today
    Lback_date_raw  = today_raw - timedelta(days=Lookback*365)
else:
    date_select = selected_date
    selected_date_raw = datetime.strptime(selected_date, '%Y-%m-%d')
    Lback_date_raw  = selected_date_raw - timedelta(days=Lookback*365)

arima_start = Lback_date_raw - timedelta(days=ARIMA_PreTrain*365)

data = yf.download(Stock, start=arima_start, end=date_select)
data['Day'] = list(range(0,data.shape[0]))
data = data.reset_index()
data = data.astype({'Volume': 'float64'}) # change datatype to float

data['open_return']         = np.nan
data['close_return']        = np.nan
data['high_return']         = np.nan
data['low_return']          = np.nan
data['vol_return']          = np.nan
data['range_return']        = np.nan
data['open_return_log']     = np.nan
data['close_return_log']    = np.nan
data['high_return_log']     = np.nan
data['low_return_log']      = np.nan
data['vol_return_log']      = np.nan
data['range_return_log']    = np.nan
data['month']               = np.nan
data['day_month']           = np.nan
data['day_week']            = np.nan

print('Creating data parms')

for i,row in data.iterrows():

    if not i:
        # skip the first row
        continue
    
    # create norm parameters
    # Let the Close price of the stock for D1 is X1
    #and that for D2 is X2. Then, close_return for D2 computed as
    #(X2 - X1)/X1 in terms of percentage. 

    #returns
    data.at[i,'open_return']    = (data.iloc[i]['Open'] - data.iloc[i-1]['Open'])/data.iloc[i-1]['Open']*100
    data.at[i,'close_return']   = (data.iloc[i]['Close'] - data.iloc[i-1]['Close'])/data.iloc[i-1]['Close']*100
    data.at[i,'high_return']    = (data.iloc[i]['High'] - data.iloc[i-1]['High'])/data.iloc[i-1]['High']*100
    data.at[i,'low_return']     = (data.iloc[i]['Low'] - data.iloc[i-1]['Low'])/data.iloc[i-1]['Low']*100
    data.at[i,'vol_return']     = (data.iloc[i]['Volume'] - data.iloc[i-1]['Volume'])/data.iloc[i-1]['Volume']*100

    ranged2                     = data.iloc[i]['High'] - data.iloc[i]['Low']
    ranged1                     = data.iloc[i-1]['High'] - data.iloc[i-1]['Low']
    data.at[i,'range_return']   = (ranged2 - ranged1)/ranged1*100

    #log returns
    data.at[i,'open_return_log']        = np.log(data.iloc[i]['Open']) - np.log(data.iloc[i-1]['Open'])
    data.at[i,'close_return_log']       = np.log(data.iloc[i]['Close']) - np.log(data.iloc[i-1]['Close'])
    data.at[i,'high_return_log']        = np.log(data.iloc[i]['High']) - np.log(data.iloc[i-1]['High'])
    data.at[i,'low_return_log']         = np.log(data.iloc[i]['Low']) - np.log(data.iloc[i-1]['Low'])
    data.at[i,'vol_return_log']         = np.log(data.iloc[i]['Volume']) - np.log(data.iloc[i-1]['Volume'])
    data.at[i,'range_return_log']       = np.log(ranged2) - np.log(ranged1)

    data.at[i,'month']      = row['Date'].month
    data.at[i,'day_month']  = row['Date'].day
    data.at[i,'day_week']   = row['Date'].weekday()

    #used for converting return later on
    data.at[i,'PreviousClose']       = data.iloc[i-1]['Close']

# delete first row or any rows with nans
data = data.dropna()
data = data.reset_index(drop=True)
 
print('Creating ARIMA model.')
# separte the arima data from the train/test data arima_history = first year
ARIMApretrain  = data.iloc[0:ARIMA_PreTrain*253][ARIMA_Predict] # 253 is the number of trading days per year
ARIMAchase     = data.iloc[ARIMA_PreTrain*253:][ARIMA_Predict]

# ARIMA model train and output
history = [x for x in ARIMApretrain]

differenced = difference(history)

#auto arima
AutoArima = pm.auto_arima(differenced, seasonal=True, m=12, suppress_warnings=True)

predictions = list()
obs         = list()

for t in range(len(ARIMAchase)):
    model = SARIMAX(differenced, order=AutoArima.order, seasonal_order=AutoArima.seasonal_order)
    #model = ARIMA(history, order=AutoArima.order)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(H)

    copy_history = history
    temp = np.array(np.zeros(H))
    i = 0
    for yhat in output:
        inverted_yhat = inverse_difference(copy_history, yhat)
        temp[i] = inverted_yhat
        copy_history.append(inverted_yhat)
        i+=1

    #yhat = predicted differenced value
    yhat = output[0]
    #predictions = yhat + history[-1]
    predictions.append(temp)
    obs.append(ARIMAchase.iloc[t])

    #update history with the actual value
    history.append(ARIMAchase.iloc[t])




arimadf = pd.DataFrame()
for day in range(0,len(predictions)):
    for t in range(0,len(predictions[0])):
        arimadf=arimadf.append({'Day':day + t, 'T+':t+1, 'Value':predictions[day][t], 'Pred/Actual':'Pred'}, ignore_index=True)

arimadf_cut = pd.DataFrame()
day_count = 0
for i,row in arimadf.iterrows():
    if (row['Day'] == day_count) & (row['T+']==1):
        for h in range(0,H):
            arimadf_cut = arimadf_cut.append(arimadf.iloc[i+h])
        day_count += H*2




#%% Post

print('Done')



true = pd.DataFrame()
true['Value'] = obs
true['Pred/Actual']='Actual'
true['T+'] = 0
true = true.reset_index(drop=False)
true = true.rename(columns={"index": "Day"})

arimadf_cut = arimadf_cut.append(true)

sns.relplot(x='Day', y='Value', hue='T+', style='Pred/Actual' , estimator=None, kind="scatter", data=arimadf_cut)


# trend_match = get_trend_match(arimadf,'Close','Actual_T+' + str(H),'ARIMA_T+' + str(H))

# arima_mae   = mean_absolute_error(obs,predictions)
# arima_mape  = mean_absolute_percentage_error(obs,predictions)

# print('ARIMA Trend Match: ' + str(trend_match) + '%')
# print('ARIMA MAE: ' + str(arima_mae))
# print('ARIMA MAPE: ' + str(arima_mape))


# for t in range(len(ARIMAchase)):
#     model = SARIMAX(history, order=AutoArima.order, seasonal_order=AutoArima.seasonal_order)
#     #model = ARIMA(history, order=AutoArima.order)
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast(H)
#     yhat = output[0]
#     predictions.append(yhat)
#     obs.append(ARIMAchase.iloc[t])
#     history.append(ARIMAchase.iloc[t])