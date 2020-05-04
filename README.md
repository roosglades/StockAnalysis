# X Day Market Forecast
 
## Background:

Market forecasting has always been a notoriously difficult task. Factors such as global macro-economincs, political, public sentiment do not follow strict patterns. Tradionally "technical factors" such as EMA (exponential moving average), RSI (relative strength index), MACD (Moving Average Convergence Divergence) and others are used to predict buys/sells for any specific stock. Recently, machine learning has shown promise in predicting future stock price performance. Machine learning used to forecast stock price coupled with traditional financial analysis could provide a more confident picture of future price movements. 

## Objective:

Leverage machine learning to provide a forecast for any given stock (on the NYSE) for any number of days in the future.

## Method:

Utilize a LSTM (Long Short Term Memory) RNN (Recurrent Neural Network) that can weight previous time sequenced inputs from the economic history and train on these hidden patterns. We will use 8 1/2 years worth of stock and global economic data to train/test the neural network's performance. 

Our performance will be measured using MAE, MAPE. However, my main criteria will be how often the trend for the forecasted days matches the actual data. 

## Data:
The data will be gathered from the following sources using mostly yahoo finance throught the yfinance python library.
- target stock: volume, high/low, range, close, adj close
- day of week, day of month, month
- sister stocks: these are stocks that are in a similar industry and standing with the target stock
- composite indices etfs: these are etfs that track major market health from the top 4 stock exchanges (NASDAQ, JPX, SSE, LSE)
- volatility index: etf tracking Chicago Board Options Exchange's CBOE Volatility Index
- currency exchange: US vs Japan, China, Europe
- Inverse fourier transform of discrete fourier tranformed closing prices
- 12 Technical Indicators: ma200, ma50, MACD etc.
- Seasonal ARIMA (AutoRegressive Integrated Moving Average) which is often used in time series prediction. We will predict closing price 

## Model:

<img width="691" alt="lstm" src="https://user-images.githubusercontent.com/43393452/80933635-8edfe180-8d92-11ea-8928-7cd9a632d197.PNG">

- Single LSTM layer
- Recurrent dropout to prevent overfitting
- Activation Function: ReLU (Rectified Linear Unit) 
- Loss: MAE (Mean Absolute Error)
- Optimizer: ADAM (Adaptive Moment Estimation)
- Node Size: Found during grid search
- Batch Size: Found during grid search
- Time Step: Found during grid search

## Analysis Flow:

1) Injest Data - 104 parameters in total
2) XGBoost (Gradient Boosting Model) that removes any irrelevant parameters (does have a non-detailed, detailed mode)
3) Log return all prices
4) Split data into train/test set
5) Scale data using standard scaler
6) Train models using grid search on node size, batch size, and time step. Target for LSTM RNN is [Y1,Y2,..Yx] 
7) De-scale data and inverse log to find predicted prices
8) Ouptut model performance on test set
9) Select best performing model based on matching trends
10) Perform horizon prediction for X number of days for selected stock
11) Output on macro and micro scale

## User Input:

<img width="397" alt="useri" src="https://user-images.githubusercontent.com/43393452/80933432-ba160100-8d91-11ea-8ee0-3875fac5d2a6.PNG">

## Required Libraries:

- yfinance
- pmdarima
- sklearn
- scipy
- pandas
- numpy
- statsmodels
- xgboost
- tensorflow
- keras
- matplotlib

## Output:

#### Stock: BAC     MAE: 1.3     MAPE: 4.9     Match Trend: 78%

Bank of America test set, 10 day prediction for the past year of trading. Trend match is based on each 10 day segment. 
<img width="580" alt="BAC_Test" src="https://user-images.githubusercontent.com/43393452/80932870-5985c480-8d8f-11ea-928a-1d312d5626a1.PNG">

Bank of America forecast for the next 10 days. 
<img width="768" alt="BAC_predict" src="https://user-images.githubusercontent.com/43393452/80932845-44a93100-8d8f-11ea-8be8-102493798760.PNG">
