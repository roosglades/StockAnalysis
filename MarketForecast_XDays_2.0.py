#%% Libraries
import os
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta

import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, linregress

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

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

# ! 10 day, 9 year lookback w 0.5 arima did 67% match
# TODO: probably need to just save the dataset, the dataprocessing is getting crazy lol
# TODO: eval, predictions, etc
# TODO: if satisfactory, grid search

#%% User Input
Stock           = 'AAPL' #ticker
SisterStock1    = 'MSFT'
SisterStock2    = 'GOOGL'
Predict         = 'Close_log'
Lookback        = 9 # years

H               = 10 #forecast horizon in days
ARIMA_PreTrain  = 0.5 # pretrain ARIMA in years 
ARIMA_Predict   = 'Close'

# Plot Folder
SaveData = 'D:/StockAnalytics/ForecastXday'

# model parms
test_split  = 0.25 # train/test split
BatchSize   = 8 # number of samples to update weights
TimeStep    = 20 # 2 months used in LSTM model
Epoch       = 100 # number of times to run through the data
Node        = 256 # number of LSTM Node

#BatchSizes  = list(range(8, 32, 8))
#TimeSteps  = list(range(20, 80, 20))
#Nodes      = list(range(32,352,64))

#predict current/past
current = False

# if current=False
selected_date = '2020-01-01'

#%% Data Prep

# extract data for analysis
DataExtract(SaveData, Stock, H, SisterStock1, SisterStock2,
            ARIMA_Predict, ARIMA_PreTrain,
            Lookback, current, selected_date, test_split)

data = pd.read_csv(SaveData + '/' + Stock + 'data.csv')

data = data.dropna()
data = data.reset_index(drop=True)

# drop anything that is not important, no real prices or anything
print('Dropping non-important parameters...')
data.to_csv(SaveData + '/rawdata.csv')
raw_data = data['Close']

keep_list = ['Volume','Range','month','day_month','day_week','log','MACD','20sd','ARIMA_Pred']
remove_list = list()
for c in range(0,len(data.columns)):
    if data.columns[c].split('_')[-1] in keep_list:
        continue
    #drop column
    remove_list.append(data.columns[c])

data = data.drop(remove_list, axis=1)
data = data.dropna()
data = data.reset_index(drop=True)

#%% Scaling/Splitting

print('Splitting Data...')
train, test = train_test_split(data, shuffle = False, test_size=test_split)

target_var  = train.columns.get_loc(Predict)

print('Scaling Data...')
#scaler = MinMaxScaler()
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train)
test_scaled  = scaler.transform(test)

#target_max = scaler.data_max_[target_var]
#target_min = scaler.data_min_[target_var]

mean    = scaler.mean_[target_var]
stdev   = scaler.scale_[target_var]

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
#pred_closing      = pred_closing*(target_max - target_min) + target_min
#actual_closing    = y_test*(target_max - target_min) + target_min
pred_closing      = pred_closing*stdev + mean
actual_closing    = y_test*stdev + mean
test_close        = raw_data.iloc[len(train)+TimeStep-1:-H].values[:]

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

trend_outcome = pd.DataFrame(columns=['PredSlope','PredTrend','TrueSlope','TrueTrend'])
trend_outcome['PredSlope']   = pred_trend_redux[:,0]
trend_outcome['PredTrend']   = pred_trend_redux[:,1]
trend_outcome['TrueSlope']   = act_trend_redux[:,0]
trend_outcome['TrueTrend']   = act_trend_redux[:,1]

trend_outcome['Match'] = np.nan
trend_outcome['Match'][trend_outcome['TrueTrend'] == trend_outcome['PredTrend']] = 1 
trend_outcome['Match'][trend_outcome['TrueTrend'] != trend_outcome['PredTrend']] = 0
trend_match = trend_outcome['Match'].sum()/trend_outcome.shape[0]*100 

mape = mean_absolute_percentage_error(act_close_redux,pred_close_redux)
mae  = mean_absolute_error(act_close_redux,pred_close_redux)

print('MAPE: ' + str(mape))
print('MAE: ' + str(mae))
print('Prediction Matches: ' + str(trend_match))

test_outcome = pd.DataFrame(columns=['Actual','Sanity Check','Predicted'])
test_outcome['Actual']          = act_close_redux
test_outcome['Predicted']       = pred_close_redux

fname = SaveData + '/tred_outcome.csv'
trend_outcome.to_csv(fname)
print('Saving Test Output: ' + fname)

#%% Results

#combine datasets
MasterDF    = pd.DataFrame(columns=['Closing Price','Actual/Predicted','Test/Train'])

#grab actual test
actualtestdf = pd.DataFrame({'Closing Price':act_close_redux,'Actual/Predicted':'Actual','Test/Train':'Test'})
MasterDF = MasterDF.append(actualtestdf)

#grab pred test
predtestdf = pd.DataFrame({'Closing Price':pred_close_redux,'Actual/Predicted':'Predicted','Test/Train':'Test'})
MasterDF = MasterDF.append(predtestdf)

#create index column to use as x axis for plot
actualtestdf    = actualtestdf.reset_index(drop=False)
predtestdf      = predtestdf.reset_index(drop=False)
MasterDF        = MasterDF.reset_index(drop=False)

print('Plotting...')
fig = plt.figure(figsize=(19.20,10.80))

ax = fig.add_subplot(111)
ax.plot(actualtestdf['index'].values[:],actualtestdf['Closing Price'].values[:], color='lightblue', linewidth=3)
for r in range(0,predtestdf.shape[0]):
    if not r:
        #grab first row
        prev_r = r
    
    elif (r == prev_r + H) :
        ax.plot(predtestdf['index'][prev_r:r].values[:],predtestdf['Closing Price'][prev_r:r].values[:], color='darkgreen', linewidth=3)
        prev_r = r

fname = SaveData + '/' + Stock + '_LSTM_B'+ str(BatchSize) + '_T' + str(TimeStep) + '_N' + str(Node) + '_M' + str(trend_match) + '_MAE' + str(mae)[:-12] + '_MAPE' + str(mape)[:-11] + '.svg'
plt.savefig(fname)
plt.close()
print('Saved Plot: ' + fname)

#%% Post

print('Done')

