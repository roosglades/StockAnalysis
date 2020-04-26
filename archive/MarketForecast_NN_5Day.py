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
from scipy.stats import pearsonr, linregress

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

import warnings
warnings.filterwarnings("ignore")

from functions import *

#%% Todo

# // TODO: get ARIMA prediction for 5 day doesn't do shit

# TODO: find optimal nn arch
# TODO: turn into rolling 5 day predition
# TODO: LSTM for 5 day prediction
# // TODO: set ARIMA as an input using only 1 day forecsast

#%% User Input
Stock           = 'AAPL' #ticker
Predict         = 'close_return_log'
Lookback        = 4 # years

H               = 5 #forecast horizon in days
ARIMA_PreTrain  = 1 # pretrain ARIMA in years 
ARIMA_Predict   = 'Close'

# Plot Folder
Plots = 'D:/StockAnalytics/Forecast5day'

# model parms
ScaleAll    = True
test_split  = 0.15 # train/test split
BatchSize   = 8 # number of samples to update weights
#BatchSizes  = list(range(8, 32, 8))
TimeStep    = 20 # 2 months used in LSTM model
#TimeSteps  = list(range(20, 80, 20))
Epoch       = 100 # number of times to run through the data
#Epoch = 10
Node        = 500 # number of LSTM Node
#Nodes      = list(range(32,352,64))

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

print('Getting technical indicators...')
data = get_technical_indicators(data,'Close')

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

predictions     = list()
obs             = list()
raw_predictions = list()
for t in range(len(chase)):
    model = SARIMAX(history_diff, order=AutoArima.order, seasonal_order=AutoArima.seasonal_order)
    #model = ARIMA(history, order=AutoArima.order)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    raw_predictions.append(yhat)
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
data['ARIMA_PredClose']     = predictions
data['ARIMA_UnConverted']   = raw_predictions

print('ARIMA prediction completed.')

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

#grid search
# for b in range(0,len(BatchSizes)):
#     BatchSize = BatchSizes[b]
#     for j in range(0,len(TimeSteps)):
#         TimeStep = TimeSteps[j]
#         for m in range(0,len(Nodes)):
#             Node = Nodes[m]

print('Creating Sequences...')

x_train, y_train    = CreateDataSequences(TimeStep, train_scaled, H, target_var)
x_test, y_test      = CreateDataSequences(TimeStep, test_scaled, H, target_var)

#%% Model

model = ShallowLSTM(Node,TimeStep,x_train.shape[2],y_train.shape[1])

print('Fitting Model..')
model.fit(x_train, y_train, epochs=Epoch, batch_size=BatchSize, verbose=2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto',
            restore_best_weights=True)])

#%% Model Eval

print('Model Testing..')
#test set
pred_closing      = model.predict(x_test)

#unscale parms
pred_closing      = pred_closing*(target_max - target_min) + target_min
actual_closing    = y_test*(target_max - target_min) + target_min
test_close        = test.iloc[TimeStep-1:-H]['Close'].values[:]

x           = np.linspace(1,pred_closing.shape[1],pred_closing.shape[1])
pred_trend  = np.zeros([pred_closing.shape[0],2])
act_trend   = np.zeros([actual_closing.shape[0],2])
for t in range(0,len(test_close)):
    current_val = test_close[t]
    for t_plus in range(0,pred_closing.shape[1]):

        pred_val    = pred_closing[t,t_plus]
        act_val     = actual_closing[t,t_plus]

        if not t_plus:
            # first point we need to use the current val
            pred_val    = np.e**pred_val*current_val
            act_val     = np.e**act_val*current_val
        else:
            # after first point we need to propogate the previous converted point
            prev_pred = pred_closing[t,t_plus-1]
            prev_act  = actual_closing[t,t_plus-1]

            pred_val    = np.e**pred_val*prev_pred
            act_val     = np.e**act_val*prev_act

        pred_closing[t,t_plus]      = pred_val
        actual_closing[t,t_plus]    = act_val

    #grabbing overall forecast trend
    pred_trend[t,0] = linregress(x, pred_closing[t]).slope
    if pred_trend[t,0] > 0:
        pred_trend[t,1] = 1
    elif pred_trend[t,0] <= 0:
        pred_trend[t,1] = 0
    
    act_trend[t,0]  = linregress(x, actual_closing[t]).slope
    if act_trend[t,0] > 0:
        act_trend[t,1] = 1
    elif act_trend[t,0] <= 0:
        act_trend[t,1] = 0

act_close_redux, pred_close_redux = list(), list()
act_trend_redux, pred_trend_redux = list(), list()
for r in range(0,len(actual_closing)):
    
    if not r:
        # first row
        act_close_redux.append(actual_closing[r])
        pred_close_redux.append(pred_closing[r])

        act_trend_redux.append(act_trend[r])
        pred_trend_redux.append(pred_trend[r])
        
        prev_r = r
    
    elif r == prev_r + H :

        act_close_redux.append(actual_closing[r])
        pred_close_redux.append(pred_closing[r])

        act_trend_redux.append(act_trend[r])
        pred_trend_redux.append(pred_trend[r])

        prev_r = r


pred_trend_redux    = np.array(pred_trend_redux)
act_trend_redux     = np.array(act_trend_redux)
act_close_redux     = np.concatenate(act_close_redux)
pred_close_redux    = np.concatenate(pred_close_redux)

test_outcome = pd.DataFrame(columns=['PredSlope','PredTrend','TrueSlope','TrueTrend', 'Actual', 'Predicted'])
test_outcome['PredSlope']   = pred_trend_redux[:,0]
test_outcome['PredTrend']   = pred_trend_redux[:,1]
test_outcome['TrueSlope']   = act_trend_redux[:,0]
test_outcome['TrueTrend']   = act_trend_redux[:,1]

test_outcome['Match'] = np.nan
test_outcome['Match'][test_outcome['TrueTrend'] == test_outcome['PredTrend']] = 1 
test_outcome['Match'][test_outcome['TrueTrend'] != test_outcome['PredTrend']] = 0
trend_match = test_outcome['Match'].sum()/test_outcome.shape[0]*100 

mape = mean_absolute_percentage_error(act_close_redux,pred_close_redux)
mae  = mean_absolute_error(act_close_redux,pred_close_redux)

print('MAPE: ' + str(mape))
print('MAE: ' + str(mae))
print('Prediction Matches: ' + str(trend_match))

fname = Plots + '/test_outcome.csv'
test_outcome.to_csv(fname)
print('Saving Test Output: ' + fname)

#%% Results

#combine datasets
MasterDF    = pd.DataFrame(columns=['Closing Price','Actual/Predicted','Test/Train'])

#grab actual test
tempdf = pd.DataFrame({'Closing Price':act_close_redux,'Actual/Predicted':'Actual','Test/Train':'Test'})
MasterDF = MasterDF.append(tempdf)

#grab pred test
tempdf = pd.DataFrame({'Closing Price':pred_close_redux,'Actual/Predicted':'Predicted','Test/Train':'Test'})
MasterDF = MasterDF.append(tempdf)

#create index column to use as x axis for plot
#MasterDF = MasterDF.reset_index(drop=True)
MasterDF = MasterDF.reset_index(drop=False)

print('Plotting...')
palette = sns.color_palette("mako_r", 2)
fig = plt.figure(figsize=(19.20,10.80))
sns.relplot(x='index', y='Closing Price', hue="Actual/Predicted", palette=palette, estimator=None, kind="line", data=MasterDF)
fname = Plots + '/' + Stock + '_LSTM_B'+ str(BatchSize) + '_T' + str(TimeStep) + '_N' + str(Node) + '_M' + str(trend_match) + '_MAE' + str(mae)[:-12] + '_MAPE' + str(mape)[:-11] + '.svg'
plt.savefig(fname)
plt.close()
print('Saved Plot: ' + fname)

#%% Post

print('Done')

