# X Day Market Forecast
 
## Background:

Market forecasting has always been a notoriously difficult task. Factors such as global macro-economincs, political, public sentiment do not follow strict patterns. Tradionally "technical factors" such as EMA (exponential moving average), RSI (relative strength index), MACD (Moving Average Convergence Divergence) and others are used to predict buys/sells for any specific stock. Lately, machine learning has shown promise in predicting future stock performance. Using a machine learning algorithm pieced together with market research and traditional methods could provide a stronger confidence in a stock's future outlook.

## Objective:

Leverage machine learning to provide a forecast for any given stock price present on the NYSE. 

## Method:

Utilize a LSTM (Long Short Term Memory) RNN (Recurrent Neural Network) that can weight previous time sequenced inputs from the economy's history and train on these hidden patterns. We will use a dataset of 8 1/2 years worth of stock and global economic data to train/test the neural network's performance. 

Our performance will be measured using a MAE, MAPE. But my main criteria will be how often the trend for the forecasted days matches the actual data. 

## Data:
The data will be gathered from the following sources using mostly yahoo finance throught the yfinance python library.
- target stock: volume, high/low, range, close, adj close
- sister stocks: these are stocks that are in a similar industry and standing with the target stock
- composite indices etfs: these are etfs that track major market health from the top 4 stock exchanges (NASDAQ, JPX, SSE, LSE)
- volatility index: etf tracking Chicago Board Options Exchange's CBOE Volatility Index
- currency exchange: US vs Japan, China, Europe
- Inverse fourier transform of discrete fourier tranformed closing price
- Technical Indicators: ma200, ma50, MACD etc.
- Seasonal ARIMA (AutoRegressive Integrated Moving Average) which is often used in time series prediction

## Analysis Flow:
1) Injest Data - 104 parameters total
2) XGBoost (Gradient Boosting Model) that removes any irrelevant parameters
3) 


## User Input:
