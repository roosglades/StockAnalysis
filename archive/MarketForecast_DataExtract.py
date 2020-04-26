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

class MyXGBClassifier(XGBClassifier):
    @property
    def coef_(self):
        return None

#%% Todo


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

#predict current/past
current = False

# if current=False
selected_date = '2020-01-01'

test_split  = 0.25

def DataExtract(SaveData,Stock,SisterStock1,SisterStock2,ARIMA_Predict,ARIMA_PreTrain,Lookback,current,selected_date,test_split):
    #%% Pre
    if not os.path.exists(SaveData):
        os.makedirs(SaveData)

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

    #Fouier Fast Transform of data
    data_FT = data[['Date', 'Close']]
    close_fft = np.fft.fft(np.asarray(data_FT['Close'].tolist()))
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 10]:
        fft_list_m10= np.copy(fft_list)
        fft_list_m10[num_:-num_]=0
        data['FFT' + str(num_) + '_Close'] = np.fft.ifft(fft_list_m10)
        data['FFT' + str(num_) + '_Close'] = data['FFT' + str(num_) + '_Close'].apply(np.absolute)
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
    plt.plot(data['Close'],  label='Real')
    plt.legend()
    plt.savefig(SaveData + '/FFT.png')
    plt.close()

    print('Getting technical indicators...')
    data = get_technical_indicators(data,'Close')

    #%% ARIMA

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
    data['ARIMA_Pred']     = predictions
    print('ARIMA prediction completed.')

    data = data.dropna()
    data = data.reset_index(drop=True)

    #%% XGBoost Feature Importance

    print('XGBOOST getting feature importance....')
    print('Predicting Horizon: ' + str(H) + ' days')

    boost_data = data

    remove_list = list()
    for c in range(0,len(boost_data.columns)):
        if 'Date' in boost_data.columns[c]:
            remove_list.append(boost_data.columns[c]) 
    boost_data = boost_data.drop(remove_list, axis=1)
    # delete first row or any rows with nans
    boost_data = boost_data.dropna()
    boost_data = boost_data.reset_index(drop=True)

    boost_target = boost_data.columns.get_loc('Close')

    #needs raw values it seems
    boosttrain, boosttest = train_test_split(boost_data, shuffle = False, test_size=test_split)

    # split data into x and y
    xgbtrain = boosttrain.iloc[:boosttrain.shape[0]-H+1]
    ygbtrain = boosttrain.iloc[H-1:][boosttrain.columns[boost_target]]

    xgbtest = boosttest.iloc[:boosttest.shape[0]-H+1]
    ygbtest = boosttest.iloc[H-1:][boosttest.columns[boost_target]]

    # fit model no training data
    model = MyXGBClassifier()
    model.fit(xgbtrain, ygbtrain)

    importances     = model.feature_importances_
    features        = xgbtrain.columns.to_list()

    featuredf = pd.DataFrame()
    featuredf['Features'] = features
    featuredf['Importances'] = importances
    featuredf=featuredf.sort_values('Importances')

    thresholds = featuredf['Importances'].to_list()

    best_mae = 999
    for thresh in thresholds:
        if thresh == 0:
            continue
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(xgbtrain)
        selected_fs=featuredf[featuredf['Importances'] >= thresh]['Features'].to_list()
        print('Num of Features: ' + str(len(selected_fs)))
        # train model
        selection_model = XGBClassifier()
        selection_model.fit(select_X_train, ygbtrain)
        # eval model
        select_X_test = selection.transform(xgbtest)
        predictions = selection_model.predict(select_X_test)
        xboost_mae = mean_absolute_error(ygbtest.values[:], np.array(predictions))
        print('XBoost MAE: ' + str(xboost_mae))

        if best_mae > xboost_mae:
            best_mae        = xboost_mae
            best_features   = selected_fs
            break

    print('XGBoost Complete')

    # grab only the important features
    if not 'Close' in best_features:
        best_features.append('Close')
    
    # take only selected features
    data = data[best_features]

    data = data.dropna()
    data = data.reset_index(drop=True)

    #%% Continue Data Creation
    print('Creating log return parms...')
    log_list = ['Open','Close','High','Low','Adj Close',
                'ma7','ma21','ma50','ma200','26ema','12ema',
                'upper_band','lower_band','ema','momentum']
    for c in range(0,len(data.columns)):
        if data.columns[c].split('_')[-1] in log_list:
            data = log_return(data,data.columns[c]) 

    data = data.dropna()
    data = data.reset_index(drop=True)
    data.to_csv(SaveData + '/' + Stock + 'data.csv')
    print('Dataset ready.')

DataExtract(SaveData,Stock,SisterStock1,SisterStock2,ARIMA_Predict,ARIMA_PreTrain,Lookback,current,selected_date,test_split)