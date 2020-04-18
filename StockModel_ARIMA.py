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
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM
# from keras.callbacks import EarlyStopping
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt


#%% Todo

# TODO: look at SARIMA

#Lessons learned:
# prediciting on just closing price - ARIMA Results
# predicting close_norm has no real effect on the outcome - ARIMA Results 2
# predicting on close_diff - ARIMA Results 3


#%% functions
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#%% User Input
Stock       = 'AAPL' #ticker
Lookback    = 4 # years 
#Predict     = 'Close'
Predict     = 'close_norm'

# Plot Folder
Plots = 'D:/StockAnalytics/ARIMA_Results2'

# model parms
test_split = 0.25 # train/test split

#ARIMA order
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 1)

P_values = range(1, 3)
Q_values = range(0, 2)
D_values = range(0, 1)

m = 12

# best match
#p=0
#d=1
#q=0

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
data['PrevClose']   = np.nan

for i,row in data.iterrows():

    if not i:
        # skip first row
        continue
    
    # create norm parameters
    # Let the Close price of the stock for D1 is X1
    #and that for D2 is X2. Then, close_norm for D2 computed as
    #(X2 - X1)/X1 in terms of percentage. 

    data.at[i,'open_norm']  = (data.iloc[i]['Open'] - data.iloc[i-1]['Open'])/data.iloc[i-1]['Open']*100
    data.at[i,'close_norm'] = (data.iloc[i]['Close'] - data.iloc[i-1]['Close'])/data.iloc[i-1]['Close']*100
    data.at[i,'high_norm']  = (data.iloc[i]['High'] - data.iloc[i-1]['High'])/data.iloc[i-1]['High']*100
    data.at[i,'low_norm']   = (data.iloc[i]['Low'] - data.iloc[i-1]['Low'])/data.iloc[i-1]['Low']*100
    data.at[i,'vol_norm']   = (data.iloc[i]['Volume'] - data.iloc[i-1]['Volume'])/data.iloc[i-1]['Volume']*100
    data.at[i,'close_diff'] = (data.iloc[i]['Close'] - data.iloc[i-1]['Close'])

    data.at[i, 'PrevClose'] = data.iloc[i-1]['Close']

    ranged2                 = data.iloc[i]['High'] - data.iloc[i]['Low']
    ranged1                 = data.iloc[i-1]['High'] - data.iloc[i-1]['Low']
    data.at[i,'range_norm'] = (ranged2 - ranged1)/ranged1*100

    data.at[i,'month']      = row['Date'].month
    data.at[i,'day_month']  = row['Date'].day
    data.at[i,'day_week']   = row['Date'].weekday()

# delete first row or any rows with nans
data = data.dropna()
print('Splitting Data...')
train, test = train_test_split(data, shuffle = False, test_size=test_split)

ytrain  = train[Predict].values
ytest   = test[Predict].values

# grid search
for p in p_values:
    for d in d_values:
        for q in q_values:
            for P in P_values:
                for D in D_values:
                    for Q in Q_values:

                        #%% Model
                        history = [x for x in ytrain]
                        predictions = list()
                        for t in range(len(ytest)):
                            #model = ARIMA(history, order=(p,d,q))
                            model = SARIMAX(history, order=(p,d,q), seasonal_order=(P,D,Q,m))
                            model_fit = model.fit(disp=0)
                            output = model_fit.forecast()
                            #yhat = output[0][0]
                            yhat = output[0]
                            predictions.append(yhat)
                            obs = ytest[t]
                            history.append(obs)

                        #%% Eval

                        if Predict == 'close_norm' or Predict == 'close_diff':
                            test['PredCloseNorm']   = predictions
                            test['ObsCloseNorm']    = ytest

                            test['ConvPredClose'] = (test['PredCloseNorm']) + test['PrevClose']
                            test['ConvObsClose']  = (test['ObsCloseNorm']) + test['PrevClose']

                            #convert
                            predictions = test['ConvPredClose'].values[:]
                            SwitchPredict = 'Close'
                        
                        else:
                            SwitchPredict = Predict


                        MasterDF = pd.DataFrame(columns=['Date',SwitchPredict,'Actual/Predicted'])

                        # Test Prediction
                        test_dates = data['Day'][len(train):]
                        tempdf = pd.DataFrame({'Date':test_dates,SwitchPredict:predictions,'Actual/Predicted':'Predicted','Test/Train':'Test'})
                        MasterDF = MasterDF.append(tempdf)

                        # Test Actual
                        tempdf = pd.DataFrame({'Date':test_dates,SwitchPredict:test[Predict].tolist(),'Actual/Predicted':'Actual','Test/Train':'Test'})
                        MasterDF = MasterDF.append(tempdf)


                        mape = mean_absolute_percentage_error(test[SwitchPredict].tolist(),predictions)
                        mae  = mean_absolute_error(test[SwitchPredict].tolist(),predictions)
                        corr, _ = pearsonr(test[SwitchPredict].tolist(), predictions)

                        # trend matches
                        test_outcome = pd.DataFrame(columns=['Day',SwitchPredict,'Pred','Actual'])

                        test_outcome['Day']         = test_dates
                        test_outcome[SwitchPredict]       = data[SwitchPredict][len(train)-1:-1].values[:]
                        test_outcome['Pred']        = predictions
                        test_outcome['Actual']      = test[SwitchPredict].tolist()

                        test_outcome['PredTrend']   = test_outcome['Pred'] - test_outcome[SwitchPredict]
                        test_outcome['PredTrend'][test_outcome['PredTrend'] <= 0]   = 0
                        test_outcome['PredTrend'][test_outcome['PredTrend'] > 0]    = 1
                        test_outcome['ActualTrend'] = test_outcome['Actual'] - test_outcome[SwitchPredict]
                        test_outcome['ActualTrend'][test_outcome['ActualTrend'] <= 0]   = 0
                        test_outcome['ActualTrend'][test_outcome['ActualTrend'] > 0]    = 1

                        test_outcome['Match'] = np.nan
                        test_outcome['Match'][test_outcome['ActualTrend'] == test_outcome['PredTrend']] = 1 
                        test_outcome['Match'][test_outcome['ActualTrend'] != test_outcome['PredTrend']] = 0
                        trend_match = test_outcome['Match'].sum()/test_outcome.shape[0]*100 

                        print('MAPE: ' + str(mape))
                        print('MAE: ' + str(mae))
                        print('Correlation: ' + str(corr))
                        print('Prediction Matches: ' + str(trend_match))

                        if not os.path.exists(Plots):
                            os.makedirs(Plots)

                        print('Plotting...')
                        palette = sns.color_palette("mako_r", 2)
                        fig = plt.figure(figsize=(19.20,10.80))
                        sns.relplot(x="Date", y=SwitchPredict, hue="Actual/Predicted",style="Test/Train", palette=palette, estimator=None, kind="line", data=MasterDF)
                        fname = Plots + '/' + Stock + '_ARIMA_O'+ str(p) + str(d) + str(q) +'_SO_' + str(P) + str(D) + str(Q) + '_M' + str(trend_match) + '_C' + str(corr) + '_MAE' + str(mae) + '_MAPE' + str(mape) + '.svg'
                        plt.savefig(fname)
                        plt.close()
                        print('Saved Plot: ' + fname)

#%% Post

print('Done')