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
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

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

#%% functions

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