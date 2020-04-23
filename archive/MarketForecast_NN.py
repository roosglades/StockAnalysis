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
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
Lookback        = 2 # years

H               = 5 #forecast horizon in days
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
Epoch       = 150 # number of times to run through the data
#Epochs     = list(range(50,550,50))
Node        = 256 # number of LSTM Node
#Nodes      = list(range(16,144,16))

#predict current/past
current = False

# if current=False
selected_date = '2020-01-01'

#%% Pre
if not os.path.exists(Plots):
    os.makedirs(Plots)

#%% Creating Dataset
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

arima_start = Lback_date_raw - timedelta(days=round(ARIMA_PreTrain*365))

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
ARIMApretrain  = data.iloc[0:round(ARIMA_PreTrain*253)][ARIMA_Predict] # 253 is the number of trading days per year
ARIMAchase     = data.iloc[round(ARIMA_PreTrain*253)-1:][ARIMA_Predict] #-1 so i can grab the difference and keep the contination

# ARIMA model train and output
history = [x for x in ARIMApretrain]
chase   = [x for x in ARIMAchase]

history_diff = difference(history).tolist()
chase_diff   = difference(chase).tolist()
chase.pop(0)

#auto arima
AutoArima = pm.auto_arima(history_diff, seasonal=True, m=12, suppress_warnings=True)

predictions = list()
obs         = list()
for t in range(len(chase)):
    model = SARIMAX(history_diff, order=AutoArima.order, seasonal_order=AutoArima.seasonal_order)
    #model = ARIMA(history, order=AutoArima.order)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(inverse_difference(history,yhat))
    obs.append(chase[t])
    history.append(chase[t])
    history_diff.append(chase_diff[t])

arima_metrics = forecast_accuracy(np.array(predictions),np.array(obs))
arima_mse = mean_squared_error(obs,predictions)

print('ARIMA MAE: ' + str(arima_metrics['mae']))
print('ARIMA MAPE: ' + str(arima_metrics['mape']))
print('ARIMA MSE: ' + str(arima_mse))

# separate data that was used to pretrain ARIMA
data = data.iloc[round(ARIMA_PreTrain*253):]

#add in ARIMA estimate
data['ARIMA_PredClose'] = predictions

print('ARIMA prediction completed.')

print('Getting technical indicators...')
data = get_technical_indicators(data,'Close')
data = data.dropna()
data = data.reset_index(drop=True)

print('Dataset ready.')
#%% Scaling/Splitting

print('Splitting Data...')
data_prep   = data.drop(['Day','Date'],axis=1)
train, test = train_test_split(data_prep, shuffle = False, test_size=test_split)

target_var  = train.columns.get_loc(Predict)

print('Scaling Data...')
scaler = MinMaxScaler()

if ScaleAll:
    train_scaled = scaler.partial_fit(train)
    test_scaled  = scaler.partial_fit(test)
    train_scaled = scaler.transform(train)
    test_scaled  = scaler.transform(test)
else:
    train_scaled = scaler.fit_transform(train)
    test_scaled  = scaler.transform(test)

target_max = scaler.data_max_[target_var]
target_min = scaler.data_min_[target_var]


#%% Create LSTM Sequences

print('Creating Sequences...')

# Create sequences of time steps for each y prediction
x_train, y_train = [], []
for i in range(TimeStep,len(train_scaled)):
    x_train.append(train_scaled[i-TimeStep:i,:])
    y_train.append(train_scaled[i,target_var])
x_train, y_train = np.array(x_train), np.array(y_train)

x_test, y_test = [], []
for i in range(TimeStep,len(test_scaled)):
    x_test.append(test_scaled[i-TimeStep:i,:])
    y_test.append(test_scaled[i,target_var])
x_test, y_test = np.array(x_test), np.array(y_test)

#%% Model

model = ShallowLSTM(Node,TimeStep,x_train.shape[2])

print('Fitting Model..')
model.fit(x_train, y_train, epochs=Epoch, batch_size=BatchSize, verbose=2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto',
            restore_best_weights=True)])

#%% Model Eval

print('Model Testing..')
#test set
pred_closing      = model.predict(x_test)
pred_closing      = np.concatenate((pred_closing[:,:]))

#unscale parms
pred_closing      = pred_closing*(target_max - target_min) + target_min
actual_closing    = y_test*(target_max - target_min) + target_min

#create test table
test_outcome = pd.DataFrame(columns=[Predict,'Close','PredReturn','ActualReturn','PredClose','ActualClose'])
test_outcome[Predict]           = test.iloc[TimeStep-1:-1][Predict].values[:]
test_outcome['Close']           = test.iloc[TimeStep-1:-1]['Close'].values[:]
test_outcome['PredReturn']      = pred_closing
test_outcome['ActualReturn']    = actual_closing
test_outcome['PredClose']       = np.e**test_outcome['PredReturn'] * test_outcome['Close']
test_outcome['ActualClose']     = np.e**test_outcome['ActualReturn'] * test_outcome['Close']

trend_match = get_trend_match(test_outcome,'Close','ActualClose','PredClose')

mape = mean_absolute_percentage_error(test_outcome['ActualClose'].values[:],test_outcome['PredClose'].values[:])
mae  = mean_absolute_error(test_outcome['ActualClose'].values[:],test_outcome['PredClose'].values[:])
corr, _ = pearsonr(test_outcome['ActualClose'].values[:],test_outcome['PredClose'].values[:])

print('MAPE: ' + str(mape))
print('MAE: ' + str(mae))
print('Correlation: ' + str(corr))
print('Prediction Matches: ' + str(trend_match))

fname = Plots + '/test_outcome.csv'
test_outcome.to_csv(fname)
print('Saving Test Output: ' + fname)

#%% Post

print('Done')

