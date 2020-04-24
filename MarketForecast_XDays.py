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

# // TODO: Data intake: grab index performances, grab currencies, etc
# TODO: log return transform all of the prices
# TODO: Keep ARIMA but have it predict the log return
# TODO: XGBoost to get feature importance, will need to prebuild the y_train y_test sets i think
# TODO: Scaling: might need to use a differenct scaler, min max might not be the thing
# TODO: Downselect only the useful features, don't need to keep everything
# TODO: train on downselected features with shallow model
# TODO: eval, predictions, etc
# TODO: if satisfactory, grid search

#%% User Input
Stock           = 'AAPL' #ticker
SisterStock1    = 'MSFT'
SisterStock2    = 'GOOGL'
Predict         = 'Close_log'
Lookback        = 4 # years

H               = 5 #forecast horizon in days
ARIMA_PreTrain  = 0.5 # pretrain ARIMA in years 
ARIMA_Predict   = 'Close_log'

# Plot Folder
Plots = 'D:/StockAnalytics/ForecastXday'

# model parms
test_split  = 0.15 # train/test split
BatchSize   = 8 # number of samples to update weights
TimeStep    = 20 # 2 months used in LSTM model
Epoch       = 100 # number of times to run through the data
Node        = 500 # number of LSTM Node

#BatchSizes  = list(range(8, 32, 8))
#TimeSteps  = list(range(20, 80, 20))
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

arima_start = (Lback_date_raw - timedelta(days=round(ARIMA_PreTrain*365))).strftime("%Y-%m-%d")

# target stock
data = yf.download(Stock, start=arima_start, end=date_select)
data = data.reset_index()
data = data.astype({'Volume': 'float64'}) # change datatype to float
data['Range'] = data['High'] - data['Low']

# sister stocks (2)
sis1data = yf.download(SisterStock1, start=arima_start, end=date_select)
sis1data = sis1data.reset_index()
sis1data = sis1data.astype({'Volume': 'float64'}) # change datatype to float
sis1data['Range'] = sis1data['High'] - sis1data['Low']
sis1data = sis1data.add_prefix('sis1_')

sis2data = yf.download(SisterStock2, start=arima_start, end=date_select)
sis2data = sis2data.reset_index()
sis2data = sis2data.astype({'Volume': 'float64'}) # change datatype to float
sis2data['Range'] = sis2data['High'] - sis2data['Low']
sis2data = sis2data.add_prefix('sis2_')

# composite indices
# powershares etf that tracks nasdaq 100
comp1data = yf.download('qqq', start=arima_start, end=date_select)
comp1data = comp1data.reset_index()
comp1data = comp1data.astype({'Volume': 'float64'}) # change datatype to float
comp1data['Range'] = comp1data['High'] - comp1data['Low']
comp1data = comp1data.add_prefix('comp1_')

# tracks the S&P 500
comp2data = yf.download('spy', start=arima_start, end=date_select)
comp2data = comp2data.reset_index()
comp2data = comp2data.astype({'Volume': 'float64'}) # change datatype to float
comp2data['Range'] = comp2data['High'] - comp2data['Low']
comp2data = comp2data.add_prefix('comp2_')

# ishares tracking japanase market
comp3data = yf.download('jpxn', start=arima_start, end=date_select)
comp3data = comp3data.reset_index()
comp3data = comp3data.astype({'Volume': 'float64'}) # change datatype to float
comp3data['Range'] = comp3data['High'] - comp3data['Low']
comp3data = comp3data.add_prefix('comp3_')

# ishares tracking chinese market
comp4data = yf.download('ashr', start=arima_start, end=date_select)
comp4data = comp4data.reset_index()
comp4data = comp4data.astype({'Volume': 'float64'}) # change datatype to float
comp4data['Range'] = comp4data['High'] - comp4data['Low']
comp4data = comp4data.add_prefix('comp4_')

# vanguard tracking european market
comp5data = yf.download('vgk', start=arima_start, end=date_select)
comp5data = comp5data.reset_index()
comp5data = comp5data.astype({'Volume': 'float64'}) # change datatype to float
comp5data['Range'] = comp5data['High'] - comp5data['Low']
comp5data = comp5data.add_prefix('comp5_')

# volatility index
# etf tracking volatility index
vixdata = yf.download('viix', start=arima_start, end=date_select)
vixdata = vixdata.reset_index()
vixdata = vixdata.astype({'Volume': 'float64'}) # change datatype to float
vixdata['Range'] = vixdata['High'] - vixdata['Low']
vixdata = vixdata.add_prefix('vix_')

# currency trade rates
# japan to usdollar 
jpnusddata = yf.download('fxy', start=arima_start, end=date_select)
jpnusddata = jpnusddata.reset_index()
jpnusddata = jpnusddata.astype({'Volume': 'float64'}) # change datatype to float
jpnusddata['Range'] = jpnusddata['High'] - jpnusddata['Low']
jpnusddata = jpnusddata.add_prefix('jpnusd_')

#china to usdollar
chinusddata = yf.download('cyb', start=arima_start, end=date_select)
chinusddata = chinusddata.reset_index()
chinusddata = chinusddata.astype({'Volume': 'float64'}) # change datatype to float
chinusddata['Range'] = chinusddata['High'] - chinusddata['Low']
chinusddata = chinusddata.add_prefix('chinusd_')

#euro to usdollar
eurousddata = yf.download('fxe', start=arima_start, end=date_select)
eurousddata = eurousddata.reset_index()
eurousddata = eurousddata.astype({'Volume': 'float64'}) # change datatype to float
eurousddata['Range'] = eurousddata['High'] - eurousddata['Low']
eurousddata = eurousddata.add_prefix('eurousd_')

#bring it all together
data = pd.concat([data, sis1data, sis2data, comp1data, 
                comp2data, comp3data, comp4data, comp5data,
                vixdata, jpnusddata,chinusddata, eurousddata], axis=1)

data['month']               = np.nan
data['day_month']           = np.nan
data['day_week']            = np.nan

for i,row in data.iterrows():
    if not i:
        # skip the first row
        continue
    data.at[i,'month']      = row['Date'].month
    data.at[i,'day_month']  = row['Date'].day
    data.at[i,'day_week']   = row['Date'].weekday()

print('Creating log return parms...')
log_list = ['Open','Close','High','Low','Adj Close']
for c in range(0,len(data.columns)):
    if data.columns[c].split('_')[-1] in log_list:
        data = log_return(data,data.columns[c]) 

print('Getting technical indicators...')
data = get_technical_indicators(data,'Close')

# delete first row or any rows with nans
data = data.dropna()
data = data.reset_index(drop=True)
 
print('Creating ARIMA model.')
# separte the arima data from the train/test data arima_history = first year
ARIMApretrain  = data.iloc[0:round(ARIMA_PreTrain*253)][ARIMA_Predict] # 253 is the number of trading days per year
ARIMAchase     = data.iloc[round(ARIMA_PreTrain*253):][ARIMA_Predict]

# ARIMA model train and output
history = [x for x in ARIMApretrain]

#auto arima
AutoArima = pm.auto_arima(history, seasonal=True, m=12, suppress_warnings=True)

# ARIMA model train and output
history = [x for x in ARIMApretrain]
predictions = list()
for t in range(len(ARIMAchase)):
    model = SARIMAX(history, order=AutoArima.order, seasonal_order=AutoArima.seasonal_order)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    history.append(ARIMAchase.iloc[t])

# separate data that was used to pretrain ARIMA
data = data.iloc[round(ARIMA_PreTrain*253):]

#add in ARIMA estimate
data['ARIMA_Pred']     = predictions
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

