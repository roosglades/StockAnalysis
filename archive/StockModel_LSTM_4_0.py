#%% Libraries
import os
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta

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
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt


#%% Todo

# // TODO: make ARIMA an input to the model
# // TODO: grab 1year of data before training data
# // TODO: make that year = history
# // TODO: then create series of data for ARIMA's prediction for the rest of the NN's set (test/train)
# // TODO: feature reduction using XGBOOST
# // TODO: try to predict close_norm, have a feeling it will be no good and will need to switch to close
# // TODO: will need to interpret ARIMA model because it will be be normalized 
# // TODO: predict returns not the actual price, can always turn return back into price
# // TODO: use log return as a feature and to predict that
# ! tests indicate that shorter train scale help optimize matching trends
# TODO: normalizers/dropout on heavier trained model. maybe we just need to dropout the data more
# TODO: Grid search and optimize the ARIMA
# TODO: check for seasonality with decompose
# TODO: use auto_arima to guess which p,d,q values are best

#! next model
# TODO: add in 5 day ARIMA prediction
# TODO: go from 1 day to 5 day prediction
# TODO: add in twitter/google data as parm

# model optimization obs:
# adding intial recurrent_dropout layer improved performance
# didn't seem to matter adding re dropout to each layer
# adding one dropout layer after input layer did not seem to improve
# trying 4 year data with recurrent dropout layer 1 ---

#%% functions
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_technical_indicators(dataset,targetparm):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset[targetparm].rolling(window=7).mean()
    dataset['ma21'] = dataset[targetparm].rolling(window=21).mean()
    
    # Create MACD
    dataset['26ema'] = dataset[targetparm].ewm(span=26, adjust=False).mean()
    dataset['12ema'] = dataset[targetparm].ewm(span=12, adjust=False).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])
    # Create Bollinger Bands
    dataset['20sd'] = dataset[targetparm].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema'] = dataset[targetparm].ewm(com=0.5).mean()
    
    # Create Momentum
    dataset['momentum'] = dataset[targetparm]-1
    
    return dataset

#%% User Input
Stock           = 'AAPL' #ticker
Predict         = 'close_return_log'
Lookback        = 2 # years

#H               = 30 #forecast horizon
ARIMA_PreTrain  = 1 # years 
ARIMA_Predict   = 'close_return_log'

# Plot Folder
Plots = 'D:/StockAnalytics/LSTM4_arch_study'

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

p = 1
d = 0
q = 2

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
    Lback_date_raw  = today_raw - timedelta(days=Lookback*253)
else:
    date_select = selected_date
    selected_date_raw = datetime.strptime(selected_date, '%Y-%m-%d')
    Lback_date_raw  = selected_date_raw - timedelta(days=Lookback*253)

arima_start = Lback_date_raw - timedelta(days=ARIMA_PreTrain*253)

data = yf.download(Stock, start=arima_start, end=selected_date)
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
data[Predict + '_T+1'] = np.nan

for i,row in data.iterrows():

    if i+1 == data.shape[0]:
        # skip the last row
        break

    data.at[i,Predict + '_T+1'] = data.iloc[i+1][Predict]

data = data.dropna()
data = data.reset_index(drop=True)
 
print('Creating ARIMA model.')
# separte the arima data from the train/test data arima_history = first year
ARIMApretrain  = data.iloc[0:ARIMA_PreTrain*253][ARIMA_Predict] # 253 is the number of trading days per year
ARIMAchase     = data.iloc[ARIMA_PreTrain*253:][ARIMA_Predict]

# ARIMA model train and output
history = [x for x in ARIMApretrain]
predictions = list()
for t in range(len(ARIMAchase)):
    model = ARIMA(history, order=(p,d,q))
    #model = SARIMAX(history, order=(p,d,q))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0][0]
    #yhat = output[0]
    predictions.append(yhat)
    history.append(ARIMAchase.iloc[t])

# data that has the ARIMA pre train removed from it
data = data.iloc[ARIMA_PreTrain*253:]

#place ARIMA output back in data
data['ARIMAGuess_' + ARIMA_Predict] = predictions

print('ARIMA model completed.')

print('Get technical indicators...')
data = get_technical_indicators(data,Predict)
data = data.dropna()
data = data.reset_index(drop=True)

print('Dataset ready.')
#%% Scale Data

print('Splitting Data...')
data_prep   = data.drop(['Day','Date'],axis=1)
train, test = train_test_split(data_prep, shuffle = False, test_size=test_split)

#%% XGBOOST Select Feature Importance
print('XGBOOST getting feature importance....')

# split data into x and y
xgbtrain = train.loc[:,train.columns != Predict + '_T+1']
ygbtrain = train.loc[:,Predict + '_T+1']

xgbtest = train.loc[:,train.columns != Predict + '_T+1']
ygbtest = train.loc[:,Predict + '_T+1']

# fit model no training data
model = XGBClassifier()
model.fit(xgbtrain, ygbtrain)

# plot feature importance
#plot_importance(model)
#fname = Plots + '/F_Importance.svg'
#plt.savefig(fname)

print('XGBoost Complete')

train       = train.loc[:,train.columns != Predict + '_T+1']
test        = train.loc[:,train.columns != Predict + '_T+1']
target_var  = train.columns.get_loc(Predict)

print('Scaling Data...')
scaler = MinMaxScaler()

if ScaleAll:
    train_scaled = scaler.fit(train)
    test_scaled  = scaler.fit(test)
    train_scaled = scaler.transform(train)
    test_scaled  = scaler.transform(test)
else:
    train_scaled = scaler.fit_transform(train)
    test_scaled  = scaler.transform(test)

target_max = scaler.data_max_[target_var]
target_min = scaler.data_min_[target_var]

#%% Create LSTM Sequences and split train/test

print('Creating Sequences...')

#grid search
# for j in range(0,len(TimeSteps)):
#     TimeStep = TimeSteps[j]
#     for m in range(0,len(Nodes)):
#         Node = Nodes[m]

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

dim_size = x_train.shape[2]

print('Creating Model...')
model = Sequential()
model.add(LSTM(units = Node, input_shape=(TimeStep,dim_size)))
model.add(Dense(1))


print('Model Created.')
model.compile(loss='mean_absolute_error', optimizer='adam')

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
test_outcome = pd.DataFrame(columns=[Predict,'PredReturn','ActualReturn','PredClose','ActualClose'])
test_outcome[Predict]           = test.iloc[TimeStep-1:-1][Predict].values[:]
test_outcome['Close']           = test.iloc[TimeStep-1:-1]['Close'].values[:]
test_outcome['PredReturn']      = pred_closing
test_outcome['ActualReturn']    = actual_closing
test_outcome['PredClose']       = np.e**test_outcome['PredReturn'] * test_outcome['Close']
test_outcome['ActualClose']     = np.e**test_outcome['ActualReturn'] * test_outcome['Close']

test_outcome['PredTrend']   = test_outcome['PredClose'] - test_outcome['Close']
test_outcome['PredTrend'][test_outcome['PredTrend'] <= 0]   = 0
test_outcome['PredTrend'][test_outcome['PredTrend'] > 0]    = 1
test_outcome['ActualTrend'] = test_outcome['ActualClose'] - test_outcome['Close']
test_outcome['ActualTrend'][test_outcome['ActualTrend'] <= 0]   = 0
test_outcome['ActualTrend'][test_outcome['ActualTrend'] > 0]    = 1

test_outcome['Match'] = np.nan
test_outcome['Match'][test_outcome['ActualTrend'] == test_outcome['PredTrend']] = 1 
test_outcome['Match'][test_outcome['ActualTrend'] != test_outcome['PredTrend']] = 0
trend_match = test_outcome['Match'].sum()/test_outcome.shape[0]*100 

mape = mean_absolute_percentage_error(test_outcome['ActualClose'].values[:],test_outcome['PredClose'].values[:])
mae  = mean_absolute_error(test_outcome['ActualClose'].values[:],test_outcome['PredClose'].values[:])
corr, _ = pearsonr(test_outcome['ActualClose'].values[:],test_outcome['PredClose'].values[:])

print('MAPE: ' + str(mape))
print('MAE: ' + str(mae))
print('Correlation: ' + str(corr))
print('Prediction Matches: ' + str(trend_match))

# #training set
# Tpred_closing      = model.predict(x_train)
# Tpred_closing      = np.concatenate((Tpred_closing[:,:]))
# Tpred_closing      = Tpred_closing*(target_max - target_min) + target_min
# Tactual_closing    = y_train*(target_max - target_min) + target_min

#%% Results

#combine datasets
MasterDF    = pd.DataFrame(columns=['Closing Price','Actual/Predicted','Test/Train'])

#grab actual train
#tempdf = pd.DataFrame({Predict:Tactual_closing,'Actual/Predicted':'Actual','Test/Train':'Train'})
#MasterDF = MasterDF.append(tempdf)

#grab pred train
#tempdf = pd.DataFrame({Predict:Tpred_closing,'Actual/Predicted':'Predicted','Test/Train':'Train'})
#MasterDF = MasterDF.append(tempdf)

#grab actual test
tempdf = pd.DataFrame({'Closing Price':test_outcome['ActualClose'].values[:],'Actual/Predicted':'Actual','Test/Train':'Test'})
MasterDF = MasterDF.append(tempdf)

#grab pred test
tempdf = pd.DataFrame({'Closing Price':test_outcome['PredClose'].values[:],'Actual/Predicted':'Predicted','Test/Train':'Test'})
MasterDF = MasterDF.append(tempdf)

#create index column to use as x axis for plot
#MasterDF = MasterDF.reset_index(drop=True)
MasterDF = MasterDF.reset_index(drop=False)


print('Plotting...')
palette = sns.color_palette("mako_r", 2)
fig = plt.figure(figsize=(19.20,10.80))
sns.relplot(x='index', y='Closing Price', hue="Actual/Predicted",style="Test/Train", palette=palette, estimator=None, kind="line", data=MasterDF)
fname = Plots + '/' + Stock + '_LSTM_B'+ str(BatchSize) + '_T' + str(TimeStep) + '_N' + str(Node) + '_M' + str(trend_match) + '_C' + str(corr) + '_MAE' + str(mae) + '_MAPE' + str(mape) + '.svg'
plt.savefig(fname)
plt.close()
print('Saved Plot: ' + fname)

#%% Post

print('Done')