#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 10:36:56 2019

@author: hrishekesh.shinde
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt

# Read data
dailyHistoricalData = pd.read_csv('Ford_Motors_Historical_data_2000.csv')

# hyperparameters
num_steps = 60
unit_size = 250
dropout_val = 0.2
num_epochs = 100
num_batch_size = 32

# Split data into training and testing sets
test_data_size = 30
train_data_size = dailyHistoricalData.shape[0]-test_data_size
testing_data = dailyHistoricalData.tail(2*num_steps)
training_data = dailyHistoricalData.head(train_data_size)

# get only the stock open values for all rows and open column which is the second column in dataset
stock_open_data = dailyHistoricalData.iloc[:, 1:2].values
training_data_open = training_data.iloc[:, 1:2].values
testing_data_open = testing_data.iloc[:, 1:2].values
testing_data_open = testing_data_open.reshape(-1, 1)

# stock data for volume
stock_volume_data = dailyHistoricalData.iloc[:, 6:].values
training_data_volume = training_data.iloc[:, 6:].values
testing_data_volume = testing_data.iloc[:, 6:].values
testing_data_volume = testing_data_volume.reshape(-1, 1)

# stock data for closing price
stock_close_data = dailyHistoricalData.iloc[:, 4:5].values
training_data_close = training_data.iloc[:, 4:5].values
testing_data_close = testing_data.iloc[:, 4:5].values
testing_data_close = testing_data_close.reshape(-1, 1)

# Feature scaling using MinMaxScaler for training data for open values
featureScaler_open = MinMaxScaler(feature_range=(0, 1))
scaled_training_data_open = featureScaler_open.fit_transform(training_data_open)
scaled_testing_data_open = featureScaler_open.transform(testing_data_open)

# Feature scaler for volume
featureScaler_volume = MinMaxScaler(feature_range=(0, 1))
scaled_training_data_volume = featureScaler_volume.fit_transform(training_data_volume)
scaled_testing_data_volume = featureScaler_volume.transform(testing_data_volume)

# Feature scaler for closing price
featureScaler_close = MinMaxScaler(feature_range=(0, 1))
scaled_training_data_close = featureScaler_close.fit_transform(training_data_close)
scaled_testing_data_close = featureScaler_close.transform(testing_data_close)

# Arrays for training and testing sets
X_train_open = []
y_train_open = []
X_test_open = []
y_test_open = []

X_train_volume = []
y_train_volume = []
X_test_volume = []
y_test_volume = []

X_train_close = []
y_train_close = []
X_test_close = []
y_test_close = []

# RNN needs a sequential data structure with num_steps which gives a single output
for i in range(num_steps, train_data_size):
    X_train_open.append(scaled_training_data_open[i-num_steps:i, 0])
    y_train_open.append(scaled_training_data_open[i, 0])
    X_train_volume.append(scaled_training_data_volume[i-num_steps:i, 0])
    y_train_volume.append(scaled_training_data_volume[i, 0])
    X_train_close.append(scaled_training_data_close[i-num_steps:i, 0])
    y_train_close.append(scaled_training_data_close[i, 0])
    
# Create sequential test data
for i in range(num_steps, num_steps+test_data_size):
    X_test_open.append(scaled_testing_data_open[i-num_steps:i, 0])
    y_test_open.append(scaled_testing_data_open[i-1, 0])
    X_test_volume.append(scaled_testing_data_volume[i-num_steps:i, 0])
    y_test_volume.append(scaled_testing_data_volume[i-1, 0])
    X_test_close.append(scaled_testing_data_close[i-num_steps:i, 0])
    y_test_close.append(scaled_testing_data_close[i-1, 0])
    
# convert to numpy array and reshape for open data
X_train_open, y_train_open = np.array(X_train_open), np.array(y_train_open)

X_train_open = np.reshape(X_train_open, (X_train_open.shape[0], X_train_open.shape[1], 1))
y_train_open = np.reshape(y_train_open, (y_train_open.shape[0], 1))

X_test_open, y_test_open = np.array(X_test_open), np.array(y_test_open)
X_test_open = np.reshape(X_test_open, (X_test_open.shape[0], X_test_open.shape[1], 1))
y_test_open = np.reshape(y_test_open, (y_test_open.shape[0], 1))

# convert to numpy array and reshape for volume data
X_train_volume, y_train_volume = np.array(X_train_volume), np.array(y_train_volume)

X_train_volume = np.reshape(X_train_volume, (X_train_volume.shape[0], X_train_volume.shape[1], 1))
y_train_volume = np.reshape(y_train_volume, (y_train_volume.shape[0], 1))

X_test_volume, y_test_volume = np.array(X_test_volume), np.array(y_test_volume)
X_test_volume = np.reshape(X_test_volume, (X_test_volume.shape[0], X_test_volume.shape[1], 1))
y_test_volume = np.reshape(y_test_volume, (y_test_volume.shape[0], 1))

# convert to numpy array and reshape for close data
X_train_close, y_train_close = np.array(X_train_close), np.array(y_train_close)

X_train_close = np.reshape(X_train_close, (X_train_close.shape[0], X_train_close.shape[1], 1))
y_train_close = np.reshape(y_train_close, (y_train_close.shape[0], 1))

X_test_close, y_test_close = np.array(X_test_close), np.array(y_test_close)
X_test_close = np.reshape(X_test_close, (X_test_close.shape[0], X_test_close.shape[1], 1))
y_test_close = np.reshape(y_test_close, (y_test_close.shape[0], 1))

# aggregate the data
X_train = np.concatenate((X_train_open, X_train_volume, X_train_close), axis=2)
y_train = np.concatenate((y_train_open, y_train_volume, y_train_close), axis=1)
X_test = np.concatenate((X_test_open, X_test_volume, X_test_close), axis=2)
y_test = np.concatenate((y_test_open, y_test_volume, y_test_close), axis=1)


# Build RNN
predictor = Sequential()
# First Layer
predictor.add(LSTM(units=unit_size, return_sequences=True, 
                   input_shape = (X_train.shape[1], 3)))
predictor.add(Dropout(dropout_val))
# Second Layer
predictor.add(LSTM(units=unit_size, return_sequences=True))
predictor.add(Dropout(dropout_val)) 
# Third Layer
predictor.add(LSTM(units=unit_size, return_sequences=True))
predictor.add(Dropout(dropout_val))
# Fourth Layer
predictor.add(LSTM(units=unit_size, return_sequences=False))
predictor.add(Dropout(dropout_val))
# Output Layer
predictor.add(Dense(units=3))
# compile the RNN
predictor.compile(optimizer='adam', loss='mean_squared_error')

# pass training set to RNN
predictor.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batch_size)

# make predictions
predicted_stock_price = predictor.predict(X_test)
y_test_open = featureScaler_open.inverse_transform(np.array(y_test[:, 0]).reshape(-1, 1))
y_pred_open = featureScaler_open.inverse_transform(np.array(predicted_stock_price.tolist())[:, 0].reshape(-1, 1))

y_test_volume = featureScaler_volume.inverse_transform(np.array(y_test[: , 1]).reshape(-1, 1))
y_pred_volume = featureScaler_volume.inverse_transform(np.array(predicted_stock_price.tolist())[:, 1].reshape(-1, 1))

y_test_close = featureScaler_close.inverse_transform(np.array(y_test[: , 2]).reshape(-1, 1))
y_pred_close = featureScaler_close.inverse_transform(np.array(predicted_stock_price.tolist())[:, 2].reshape(-1, 1))

# Visualize the results
plt.plot(y_pred_open, color = 'red', label='Predicted Open Price')
plt.plot(y_test_open, color = 'blue', label='Actual Open Price')
plt.title('Prediction using RNN')
plt.xlabel('Date')
plt.ylabel('Stock opening price')
plt.legend()
plt.savefig('open.png')
plt.show()

plt.plot(y_pred_volume, color = 'red', label='Predicted Volume')
plt.plot(y_test_volume, color = 'blue', label='Actual Volume')
plt.title('Prediction using RNN')
plt.xlabel('Date')
plt.ylabel('Stock Volume')
plt.legend()
plt.savefig('volume.png')
plt.show()

plt.plot(y_pred_close, color = 'red', label='Predicted Close Price')
plt.plot(y_test_close, color = 'blue', label='Actual Close Price')
plt.title('Prediction using RNN')
plt.xlabel('Date')
plt.ylabel('Stock closing price')
plt.legend()
plt.savefig('close.png')
plt.show()