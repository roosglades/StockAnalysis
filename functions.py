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

class MyXGBClassifier(XGBClassifier):
    @property
    def coef_(self):
        return None

#%% functions

def DataExtract(SaveData,Stock,Horizon,SisterStock1,
                SisterStock2,ARIMA_Predict,ARIMA_PreTrain,
                Lookback,current,selected_date,test_split,fdetail):
    #%% Pre
    if not os.path.exists(SaveData):
        os.makedirs(SaveData)
    
    folder  = SaveData + '/Data/'
    if not os.path.exists(folder):
        os.makedirs(folder)

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
    plt.savefig(folder + 'FFT.png')
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
    print('Predicting Horizon: ' + str(Horizon) + ' days')

    data.to_csv(folder + Stock + 'RawData.csv')
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
    xgbtrain = boosttrain.iloc[:boosttrain.shape[0]-Horizon+1]
    ygbtrain = boosttrain.iloc[Horizon-1:][boosttrain.columns[boost_target]]

    xgbtest = boosttest.iloc[:boosttest.shape[0]-Horizon+1]
    ygbtest = boosttest.iloc[Horizon-1:][boosttest.columns[boost_target]]

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
        if (thresh == 0):
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
            
            if fdetail:
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
    data.to_csv(folder + Stock + 'data.csv')
    print('Dataset ready.')

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse,
            'corr':corr, 'minmax':minmax})

def TrainEvalModel(Stock,SaveData,Train,Test,TargetVar,Horizon,BatchSize,TimeStep,Node,Epoch,raw_data,stdev,mean):

    print('Creating Sequences...')
    x_train, y_train    = CreateDataSequences(TimeStep, Train, Horizon, TargetVar)
    x_test, y_test      = CreateDataSequences(TimeStep, Test, Horizon, TargetVar)
    XPredict            = CreateDataSequence(TimeStep,Test)

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
    pred_closing      = pred_closing*stdev + mean
    actual_closing    = y_test*stdev + mean
    test_close        = raw_data.iloc[len(Train)+TimeStep-1:-Horizon].values[:]

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
        
        elif r == prev_r + Horizon :

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

    fname   = 'trend_outcome.csv'
    folder  = SaveData + '/Data/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    trend_outcome.to_csv(folder + fname)
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
        
        elif (r == prev_r + Horizon) :
            ax.plot(predtestdf['index'][prev_r:r].values[:],predtestdf['Closing Price'][prev_r:r].values[:], color='darkgreen', linewidth=3)
            prev_r = r

    fname = SaveData + '/Plots/' + Stock + '_LSTM_B'+ str(BatchSize) + '_T' + str(TimeStep) + '_N' + str(Node) + '_M' + str(trend_match) + '_MAE' + str(mae)[:-12] + '_MAPE' + str(mape)[:-11] + '.svg'
    plt.savefig(fname)
    plt.close()
    print('Saved Plot: ' + fname)

    return trend_match, model, XPredict


#gets the trend match for a pandas dataframe with predicted true and current price columns
def get_trend_match(dataset, current_col, true_col, predicted_col):
    
    dataset['TrueTrend'] = np.nan
    dataset['TrueTrend'] = dataset[true_col]-dataset[current_col]
    dataset['TrueTrend'][dataset['TrueTrend'] <= 0]      = 0
    dataset['TrueTrend'][dataset['TrueTrend'] > 0]       = 1

    dataset['PredTrend'] = np.nan
    dataset['PredTrend'] = dataset[predicted_col]-dataset[current_col]
    dataset['PredTrend'][dataset['PredTrend'] <= 0]      = 0
    dataset['PredTrend'][dataset['PredTrend'] > 0]       = 1
    
    dataset['Match'] = np.nan
    dataset['Match'][dataset['TrueTrend'] == dataset['PredTrend']] = 1 
    dataset['Match'][dataset['TrueTrend'] != dataset['PredTrend']] = 0
    trend_match = dataset['Match'].sum()/dataset.shape[0]*100 

    return trend_match

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_technical_indicators(dataset,targetparm):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset[targetparm].rolling(window=7).mean()
    dataset['ma21'] = dataset[targetparm].rolling(window=21).mean()
    dataset['ma50'] = dataset[targetparm].rolling(window=50).mean()
    dataset['ma200'] = dataset[targetparm].rolling(window=200).mean()

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

def log_return(dataset,transform_parm):
    dataset[transform_parm + '_log'] = np.log(dataset[transform_parm]) - np.log(dataset[transform_parm].shift(1))
    return dataset
    
def ShallowLSTM(HiddenNodes,TimeStep,input_size,output_size):
    model = Sequential()
    model.add(LSTM(units = HiddenNodes, activation='relu', recurrent_dropout=0.2, input_shape=(TimeStep,input_size)))
    model.add(Dense(output_size,activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model
    #M42.22222222222222_C0.9915_MAE2.674_MAPE1.10933

def ShallowLSTM_proto(HiddenNodes,TimeStep,input_size,output_size):
    model = Sequential()
    model.add(LSTM(units = HiddenNodes, activation='relu', recurrent_dropout=0.2, input_shape=(TimeStep,input_size)))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(output_size,activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model
    
def CreateDataSequences(TimeStep,data,Horizon,target_variable):
    x_new, y_new = [], []
    for i in range(TimeStep,len(data)):
        if i+Horizon == len(data)+1:
            break
        ytemp = []
        x_new.append(data[i-TimeStep:i,:])
        for h in range(0,Horizon):
            ytemp.append(data[i+h,target_variable])
        y_new.append(ytemp)
    x_new, y_new = np.array(x_new), np.array(y_new)

    return x_new, y_new

def CreateDataSequence(TimeStep,data):
    x_new = []
    for i in range(TimeStep,len(data)):
        x_new.append(data[i-TimeStep:i,:])
    #grab last point
    x_new = np.array(x_new[-2:-1])
    return x_new



# def create_generator():
#     generator = Sequential()
    
#     generator.add(Dense(256, input_dim=noise_dim))
#     generator.add(LeakyReLU(0.2))

#     generator.add(Dense(512))
#     generator.add(LeakyReLU(0.2))

#     generator.add(Dense(1024))
#     generator.add(LeakyReLU(0.2))

#     generator.add(Dense(img_rows*img_cols*channels, activation='tanh'))
    
#     generator.compile(loss='binary_crossentropy', optimizer=optimizer)
#     return generator

# def create_discriminator():
#     discriminator = Sequential()
     
#     discriminator.add(Dense(1024, input_dim=img_rows*img_cols*channels))
#     discriminator.add(LeakyReLU(0.2))

#     discriminator.add(Dense(512))
#     discriminator.add(LeakyReLU(0.2))

#     discriminator.add(Dense(256))
#     discriminator.add(LeakyReLU(0.2))
    
#     discriminator.add(Dense(1, activation='sigmoid'))
    
#     discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
#     return discriminator


# # how to implement
# discriminator = create_descriminator()
# generator = create_generator()

# discriminator.trainable = False

# gan_input = Input(shape=(noise_dim,))
# fake_image = generator(gan_input)

# gan_output = discriminator(fake_image)

# gan = Model(gan_input, gan_output)
# gan.compile(loss='binary_crossentropy', optimizer=optimizer)


# # training the model

# for epoch in range(epochs):
#     for batch in range(steps_per_epoch):
#         noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
#         fake_x = generator.predict(noise)
#         real_x = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
#         x = np.concatenate((real_x, fake_x))

#         disc_y = np.zeros(2*batch_size)
#         disc_y[:batch_size] = 0.9

#         # train discriminator
#         d_loss = discriminator.train_on_batch(x, disc_y)

#         #train the generator
#         y_gen = np.ones(batch_size)
#         g_loss = gan.train_on_batch(noise, y_gen)