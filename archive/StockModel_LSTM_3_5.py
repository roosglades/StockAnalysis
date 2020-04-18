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
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt


#%% Todo

# // TODO: find some optimization insights from grid search
# // TODO: add in month, day of week, day of month, range norm (diff between high and low), close norm, open norm, high norm, low norm
# // TODO: use training data for past 3 years, test for latest year
# // TODO: predict close norm
# // TODO: make "match parm", market went up, did prediction go up or down from last value
# // TODO: combine train scaled and test scaled and then create sequences. therefore  you have no gap
# // TODO: make forecast models with ARIMA and SARIMA
# TODO: make ARIMA and input to the model
# TODO: 
# TODO: feature reduction using XGBOOST
# TODO: go from 1 day to 5 day prediction
# // TODO: MAPE error and correlation for metrics
# // TODO: ADAM optimizer and MAE loss function
# TODO: add in twitter/google data as parm
# TODO: look into SARIMA peformance vs. ARIMA performance

# positive close_norm means postive trend for the next day, neg opposite etc

#%% functions
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#%% User Input
Stock       = 'AAPL' #ticker
Lookback    = 4 # years 
H           = 30 #forecast horizon
Predict     = 'Close'

# Plot Folder
Plots = 'D:/StockAnalytics/LSTM3_5_Results'

# model parms
test_split = 0.25 # train/test split
BatchSize  = 8 # number of samples to update weights
#BatchSizes  = list(range(8, 32, 8))
TimeStep  = 50 # 2 months used in LSTM model
#TimeSteps  = list(range(10, 100, 10))
Epoch     = 100 # number of times to run through the data
#Epochs     = list(range(50,550,50))
Node      = 16 # number of LSTM Node
#Nodes      = list(range(16,144,16))

#predict current/past
current = True

# if current=False
selected_date = '2020-01-01'

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

Lback_date      = Lback_date_raw.strftime("%Y-%m-%d")

data = yf.download(Stock, start=Lback_date, end=selected_date)
data['Day'] = list(range(0,data.shape[0]))
data = data.reset_index()
data = data.astype({'Volume': 'float64'}) # change datatype to float

# drop non-important parms
#data = data.drop(['Adj Close'],axis=1)

data['open_norm']   = np.nan
data['close_norm']  = np.nan
data['high_norm']   = np.nan
data['low_norm']    = np.nan
data['vol_norm']    = np.nan
data['range_norm']  = np.nan
data['month']       = np.nan
data['day_month']   = np.nan
data['day_week']    = np.nan

for i,row in data.iterrows():

    if not i:
        # skip first row
        continue
    
    # create norm parameters
    # Let the Close price of the stock for D1 is X1
    #and that for D2 is X2. Then, close_norm for D2 computed as
    #(X2 - X1)/X1 in terms of percentage. 

    data.at[i,'open_norm'] = (data.iloc[i]['Open'] - data.iloc[i-1]['Open'])/data.iloc[i-1]['Open']*100
    data.at[i,'close_norm'] = (data.iloc[i]['Close'] - data.iloc[i-1]['Close'])/data.iloc[i-1]['Close']*100
    data.at[i,'high_norm'] = (data.iloc[i]['High'] - data.iloc[i-1]['High'])/data.iloc[i-1]['High']*100
    data.at[i,'low_norm'] = (data.iloc[i]['Low'] - data.iloc[i-1]['Low'])/data.iloc[i-1]['Low']*100
    data.at[i,'vol_norm'] = (data.iloc[i]['Volume'] - data.iloc[i-1]['Volume'])/data.iloc[i-1]['Volume']*100

    ranged2                 = data.iloc[i]['High'] - data.iloc[i]['Low']
    ranged1                 = data.iloc[i-1]['High'] - data.iloc[i-1]['Low']
    data.at[i,'range_norm'] = (ranged2 - ranged1)/ranged1*100

    data.at[i,'month']      = row['Date'].month
    data.at[i,'day_month']  = row['Date'].day
    data.at[i,'day_week']   = row['Date'].weekday()

# delete first row or any rows with nans
data = data.dropna()

#remove all but important information
#data = data[['Day','Date','open_norm','close_norm','high_norm','low_norm','range_norm','vol_norm','month','day_month','day_week']]

#%% Transform Data

# because price for most stocks always continues to rise, the values will always be outside the bounds 
# and that is why the model is not predicting values outside those bounds well enough.
print('Scaling Data...')

data_prep = data.drop(['Day','Date'],axis=1)

scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(data_prep)

target_var = data_prep.columns.get_loc(Predict)
target_max = scaler.data_max_[target_var]
target_min = scaler.data_min_[target_var]

print('Creating Sequences...')

y_data_dates = list()
x_data, y_data = [], []
for i in range(TimeStep,len(data_scaled)):
    x_data.append(data_scaled[i-TimeStep:i,:])
    y_data.append(data_scaled[i,target_var])

    date=data.iloc[i]['Day']
    y_data_dates.append(date)
x_data, y_data = np.array(x_data), np.array(y_data)

print('Splitting Data...')
x_train, x_test = train_test_split(x_data, shuffle = False, test_size=test_split)
y_train, y_test = train_test_split(y_data, shuffle = False, test_size=test_split)

#grid search
# for i in range(0,len(BatchSizes)):
#     BatchSize=BatchSizes[i]
#     for j in range(0,len(TimeSteps)):
#         TimeStep = TimeSteps[j]
#         for m in range(0,len(Nodes)):
#                 Node = Nodes[m]
                
#%% Model

dim_size = x_train.shape[2]

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units = Node, return_sequences=True, input_shape=(TimeStep,dim_size)))
model.add(Dropout(0.2))
model.add(LSTM(units = int(Node/8), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = int(Node/8), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = int(Node/2)))
model.add(Dropout(0.2))
model.add(Dense(1))

print('Model Created.')
model.compile(loss='mean_absolute_error', optimizer='adam')

print('Fitting Model..')
model.fit(x_train, y_train, epochs=Epoch, batch_size=BatchSize, verbose=2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto',
            restore_best_weights=True)])


#%% Model Eval

#test set
pred_closing      = model.predict(x_test)
pred_closing      = np.concatenate((pred_closing[:,:]))
pred_closing      = pred_closing*(target_max - target_min) + target_min
actual_closing    = y_test*(target_max - target_min) + target_min

#create test table
test_outcome = pd.DataFrame(columns=['Day',Predict,'Pred','Actual'])
test_outcome['Day']     = data.iloc[TimeStep-1+len(x_train):-1]['Day'].values[:]
test_outcome[Predict]   = data.iloc[TimeStep-1+len(x_train):-1][Predict].values[:]
test_outcome['Pred']    = pred_closing
test_outcome['Actual']  = actual_closing

test_outcome['PredTrend']   = test_outcome['Pred'] - test_outcome[Predict]
test_outcome['PredTrend'][test_outcome['PredTrend'] <= 0]   = 0
test_outcome['PredTrend'][test_outcome['PredTrend'] > 0]    = 1
test_outcome['ActualTrend'] = test_outcome['Actual'] - test_outcome[Predict]
test_outcome['ActualTrend'][test_outcome['ActualTrend'] <= 0]   = 0
test_outcome['ActualTrend'][test_outcome['ActualTrend'] > 0]    = 1

test_outcome['Match'] = np.nan
test_outcome['Match'][test_outcome['ActualTrend'] == test_outcome['PredTrend']] = 1 
test_outcome['Match'][test_outcome['ActualTrend'] != test_outcome['PredTrend']] = 0
trend_match = test_outcome['Match'].sum()/test_outcome.shape[0]*100 

mape = mean_absolute_percentage_error(actual_closing,pred_closing)
mae  = mean_absolute_error(actual_closing,pred_closing)
#corr = np.corrcoef(actual_closing, pred_closing)[0, 1]
corr, _ = pearsonr(actual_closing, pred_closing)

print('MAPE: ' + str(mape))
print('MAE: ' + str(mae))
print('Correlation: ' + str(corr))
print('Prediction Matches: ' + str(trend_match))

#training set
Tpred_closing      = model.predict(x_train)
Tpred_closing      = np.concatenate((Tpred_closing[:,:]))
Tpred_closing      = Tpred_closing*(target_max - target_min) + target_min
Tactual_closing    = y_train*(target_max - target_min) + target_min

#%% Results

#combine datasets
MasterDF    = pd.DataFrame(columns=['Date', Predict,'Actual/Predicted','Test/Train'])

train_dates = data.iloc[TimeStep-1:TimeStep-1+len(x_train)]['Day'].values[:]
#grab actual train
tempdf = pd.DataFrame({'Date':train_dates,Predict:Tactual_closing,'Actual/Predicted':'Actual','Test/Train':'Train'})
MasterDF = MasterDF.append(tempdf)

#grab pred train
tempdf = pd.DataFrame({'Date':train_dates,Predict:Tpred_closing,'Actual/Predicted':'Predicted','Test/Train':'Train'})
MasterDF = MasterDF.append(tempdf)

#grab actual test
tempdf = pd.DataFrame({'Date':test_outcome['Day'],Predict:actual_closing,'Actual/Predicted':'Actual','Test/Train':'Test'})
MasterDF = MasterDF.append(tempdf)

#grab pred test
tempdf = pd.DataFrame({'Date':test_outcome['Day'],Predict:pred_closing,'Actual/Predicted':'Predicted','Test/Train':'Test'})
MasterDF = MasterDF.append(tempdf)

if not os.path.exists(Plots):
    os.makedirs(Plots)

print('Plotting...')
palette = sns.color_palette("mako_r", 2)
fig = plt.figure(figsize=(19.20,10.80))
sns.relplot(x="Date", y=Predict, hue="Actual/Predicted",style="Test/Train", palette=palette, estimator=None, kind="line", data=MasterDF)
fname = Plots + '/' + Stock + '_LSTM_B'+ str(BatchSize) + '_T' + str(TimeStep) + '_E' + str(Epoch) + '_N' + str(Node) + '_M' + str(trend_match) + '_C' + str(corr) + '_MAE' + str(mae) + '_MAPE' + str(mape) + '.svg'
plt.savefig(fname)
plt.close()
print('Saved Plot: ' + fname)

#%% Post

print('Done')