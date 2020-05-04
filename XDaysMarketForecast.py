#%% Libraries
import os
import pandas as pd
import numpy as np
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
from keras.models import load_model

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from functions import *

#%% User Input
Stock           = 'BAC'    # target stock
Horizon         = 10        # forecast horizon in days
SisterStock1    = 'JPM'    # familiar stock to target stock
SisterStock2    = 'WFC'   # familiar stock to target stock
ExtractData     = True      # Do we need to extract data?
TrainModel      = True      # Do we need to train a model?
current         = True      # Use today's date as prediction date

# if current=False
selected_date = '2020-01-01'

# Folder for plots & data
SaveData = 'D:/StockAnalytics/BAC'

#%% Model HyperParms

Predict         = 'Close_log'
Lookback        = 8.5 # years to grab data
ARIMA_PreTrain  = 0.5 # pretrain ARIMA in years 
ARIMA_Predict   = 'Close'

test_split  = 0.15 # train/test split
Epoch       = 80 # number of times to run through the data

BatchSizes  = [8,16]
TimeSteps   = [20,40,60]
Nodes       = [160,256,384,448]

#%% Pre

if not os.path.exists(SaveData):
    os.makedirs(SaveData)

folder  = SaveData + '/Data/'
if not os.path.exists(folder):
    os.makedirs(folder)

folder = SaveData + '/Plots/'
if not os.path.exists(folder):
    os.makedirs(folder)

folder = SaveData + '/Models/'
if not os.path.exists(folder):
    os.makedirs(folder)

#%% Data Prep

if ExtractData:
    # extract data for analysis
    DataExtract(SaveData, Stock, Horizon, SisterStock1, SisterStock2,
                ARIMA_Predict, ARIMA_PreTrain,
                Lookback, current, selected_date, test_split,1)

data = pd.read_csv(SaveData + '/Data/' + Stock + 'data.csv')

data = data.dropna()
data = data.reset_index(drop=True)

# drop anything that is not important, no real prices or anything
print('Dropping non-important parameters...')
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
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train)
test_scaled  = scaler.transform(test)

mean    = scaler.mean_[target_var]
stdev   = scaler.scale_[target_var]

#%% Create LSTM Sequences

if TrainModel:
    #grid search
    prev_match = 0
    for b in range(0,len(BatchSizes)):
        BatchSize = BatchSizes[b]
        for j in range(0,len(TimeSteps)):
            TimeStep = TimeSteps[j]
            for m in range(0,len(Nodes)):
                Node = Nodes[m]

                trend_match, model, NextPredict    = TrainEvalModel(Stock,SaveData,train_scaled, test_scaled, target_var,
                                                    Horizon, BatchSize, TimeStep, Node, Epoch,
                                                    raw_data, stdev, mean)

                folder  = SaveData + '/Models/'
                if not os.path.exists(folder):
                    os.makedirs(folder)

                if trend_match > prev_match:
                    # Save model file
                    model.save(folder + Stock + '_model.h5')
                    # Save trend_match
                    best_trend = trend_match
                    # Save XPredict
                    np.save(folder + Stock + '_Xdata.npy',NextPredict)
                    prev_match = trend_match

# Load Best Model
model       = load_model(folder + Stock + '_model.h5')
XPredict    = np.load(folder + Stock + '_Xdata.npy')

# Predict X
YPredict    = model.predict(XPredict)
YPredict    = np.concatenate(YPredict*stdev + mean) # un-scale the data

last_val = raw_data.values[-1]
for t_plus in range(0,YPredict.shape[0]):

    pred_val = YPredict[t_plus]
    if not t_plus:
        # first point we need to use the current val
        pred_val    = np.e**pred_val*last_val
    else:
        # after first point we need to propogate the previous converted point
        prev_pred   = YPredict[t_plus-1]
        pred_val    = np.e**pred_val*prev_pred

    YPredict[t_plus]      = pred_val

raw_data = raw_data.reset_index(drop=False)

next_idx                = raw_data['index'].values[-1] + 1
YPredictdf              = pd.DataFrame()
YPredictdf['Predicted'] = YPredict
YPredictdf              = YPredictdf.reset_index(drop=False)
YPredictdf['index']     = YPredictdf['index'] + next_idx

# Plots
print('Plotting...')
fig, (ax1, ax2) = plt.subplots(2,figsize=(19.20,10.80))

ax1.plot(raw_data['index'].values[-253:],raw_data['Close'].values[-253:], color='black', linewidth=3)
ax1.plot(YPredictdf['index'].values[:],YPredictdf['Predicted'].values[:], color='green', linewidth=3)

ax2.plot(raw_data['index'].values[-60:],raw_data['Close'].values[-60:], color='black', linewidth=3)
ax2.plot(YPredictdf['index'].values[:],YPredictdf['Predicted'].values[:], color='green', linewidth=3)


fname = SaveData + '/Models/' + 'Predict_' + str(Horizon) +'Days_'+ Stock + '.svg'
plt.savefig(fname)
plt.close()
print('Saved Plot: ' + fname)
    
#%% Post

print('Done')

