#%% Libraries
import os
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt


#%% Todo

# // TODO: find some optimization insights from grid search
# TODO: add in month, day of week, day of month, range norm (diff between high and low), close norm, open norm, high norm, low norm
# TODO: use training data for past 3 years, test for latest year
# TODO: predict close norm
# TODO: go from 1 day to 5 day prediction
# TODO: MAPE error and correlation for metrics
# TODO: ADAM optimizer and MAE loss function
# TODO: add in twitter/google data as parm

# Let the Close price of the stock for D1 is X1
#and that for D2 is X2. Then, close_norm for D2 computed as
#(X2 - X1)/X1 in terms of percentage. 

# positive close_norm means postive trend for the next day, neg opposite etc


#%% User Input
Stock       = 'AAPL' #ticker
Lookback    = 5 # years 
H           = 30 #forecast horizon
Predict     = 'Adj Close'

# Plot Folder
Plots = 'D:/StockAnalytics/LSTM_Results'

# model parms
test_split = 0.3 # train/test split
BatchSize  = 60 # number of samples to update weights
#BatchSizes  = list(range(8, 72, 8))
TimeStep  = 10 # 2 months used in LSTM model
#TimeSteps  = list(range(5, 95, 5))
Epoch     = 80 # number of times to run through the data
#Epochs     = list(range(50,550,50))
Node      = 20 # number of LSTM Node
#Nodes      = list(range(5, 55 ,5))

#%% Grabbing Data
print('Grabbing Stock Market Data...')
today_raw       = datetime.today()
today           = today_raw.strftime("%Y-%m-%d")

Lback_date_raw  = today_raw - timedelta(days=Lookback*365)
Lback_date      = Lback_date_raw.strftime("%Y-%m-%d")

data = yf.download(Stock, start=Lback_date, end=today)
data['Day'] = list(range(0,data.shape[0]))
data = data.reset_index()
data = data.astype({'Volume': 'float64'}) # change datatype to float


#%% Transform Data
print('Splitting Data...')

train, test = train_test_split(data, shuffle = False, test_size=test_split)

traindays           = pd.DataFrame(columns=['Date','Day'])
traindays['Day']    = train.Day
train               = train.drop('Day', axis=1)
traindays['Date']   = train.Date
train               = train.drop('Date', axis=1)

testdays           = pd.DataFrame(columns=['Date','Day'])
testdays['Day']    = test.Day
test               = test.drop('Day', axis=1)
testdays['Date']   = test.Date
test               = test.drop('Date', axis=1)

print('Scaling Data...')
scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train)
test_scaled  = scaler.transform(test)

target_var = train.columns.get_loc(Predict)
target_max = scaler.data_max_[target_var]
target_min = scaler.data_min_[target_var]

#%% Model

#input: open close high low adj close volume
#output: adj close

#grid search
for i in range(0,len(BatchSizes)):
    BatchSize=BatchSizes[i]
    for j in range(0,len(TimeSteps)):
        TimeStep = TimeSteps[j]
        for k in range(0,len(Epochs)):
            Epoch = Epochs[k]
            for m in range(0,len(Nodes)):
                Node = Nodes[m]
                
                print('Creating Sequences...')

                # Create sequences of time steps for each y prediction
                y_train_dates = list()
                y_test_dates  = list()

                x_train, y_train = [], []
                for i in range(TimeStep,len(train_scaled)):
                    x_train.append(train_scaled[i-TimeStep:i,:])
                    y_train.append(train_scaled[i,target_var])
                    #date=traindays.iloc[i]['Date'].strftime('%Y-%m-%d')
                    date=traindays.iloc[i]['Day']
                    y_train_dates.append(date)
                x_train, y_train = np.array(x_train), np.array(y_train)

                x_test, y_test = [], []
                for i in range(TimeStep,len(test_scaled)):
                    x_test.append(test_scaled[i-TimeStep:i,:])
                    y_test.append(test_scaled[i,target_var])
                    #date=testdays.iloc[i]['Date'].strftime('%Y-%m-%d')
                    date=testdays.iloc[i]['Day']
                    y_test_dates.append(date)
                x_test, y_test = np.array(x_test), np.array(y_test)

                dim_size = x_train.shape[2]

                # create and fit the LSTM network
                model = Sequential()
                model.add(LSTM(units = Node, return_sequences=True, input_shape=(TimeStep,dim_size)))
                model.add(Dropout(0.2))
                model.add(LSTM(units = Node, return_sequences = True))
                model.add(Dropout(0.2))
                model.add(LSTM(units = Node, return_sequences = True))
                model.add(Dropout(0.2))
                model.add(LSTM(units = Node))
                model.add(Dropout(0.2))
                model.add(Dense(1))

                print('Model Created.')
                model.compile(loss='mean_squared_error', optimizer='adam')

                print('Fitting Model..')
                model.fit(x_train, y_train, epochs=Epoch, batch_size=BatchSize, verbose=2)


                #%% Model Eval

                #test set
                pred_closing_price      = model.predict(x_test)
                pred_closing_price      = np.concatenate((pred_closing_price[:,:]))
                pred_closing_price      = pred_closing_price*(target_max - target_min) + target_min
                actual_closing_price    = y_test*(target_max - target_min) + target_min

                errors = abs(pred_closing_price - actual_closing_price) / actual_closing_price

                print('Average Percent Error: ' + str(np.average(errors)*100))
                print('Minimum Percent Error: ' + str(np.min(errors)*100))
                print('Maximum Percent Error: ' + str(np.max(errors)*100))


                #training set
                Tpred_closing_price      = model.predict(x_train)
                Tpred_closing_price      = np.concatenate((Tpred_closing_price[:,:]))
                Tpred_closing_price      = Tpred_closing_price*(target_max - target_min) + target_min
                Tactual_closing_price    = y_train*(target_max - target_min) + target_min

                #%% Results

                #combine datasets
                MasterDF    = pd.DataFrame(columns=['Date','Adj Close','Actual/Predicted','Test/Train'])

                #grab actual train
                tempdf = pd.DataFrame({'Date':y_train_dates,'Adj Close':Tactual_closing_price,'Actual/Predicted':'Actual','Test/Train':'Train'})
                MasterDF = MasterDF.append(tempdf)

                #grab pred train
                tempdf = pd.DataFrame({'Date':y_train_dates,'Adj Close':Tpred_closing_price,'Actual/Predicted':'Predicted','Test/Train':'Train'})
                MasterDF = MasterDF.append(tempdf)

                #grab actual test
                tempdf = pd.DataFrame({'Date':y_test_dates,'Adj Close':actual_closing_price,'Actual/Predicted':'Actual','Test/Train':'Test'})
                MasterDF = MasterDF.append(tempdf)

                #grab pred test
                tempdf = pd.DataFrame({'Date':y_test_dates,'Adj Close':pred_closing_price,'Actual/Predicted':'Predicted','Test/Train':'Test'})
                MasterDF = MasterDF.append(tempdf)

                print('Plotting...')
                palette = sns.color_palette("mako_r", 2)
                fig = plt.figure(figsize=(19.20,10.80))
                sns.relplot(x="Date", y="Adj Close", hue="Actual/Predicted",style="Test/Train", palette=palette, estimator=None, kind="line", data=MasterDF)
                fname = Plots + '/' + Stock + '_LSTM_B'+ str(BatchSize) + '_T' + str(TimeStep) + '_E' + str(Epoch) + '_N' + str(Node) + '.svg'
                plt.savefig(fname)
                plt.close()
                print('Saved Plot: ' + fname)

#%% Post

print('Done')