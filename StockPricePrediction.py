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
unit_size = 50
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

# Feature scaling using MinMaxScaler for training data for open values
featureScaler = MinMaxScaler(feature_range=(0, 1))
scaled_training_data = featureScaler.fit_transform(training_data_open)
scaled_testing_data = featureScaler.transform(testing_data_open)

# Arrays for training and testing sets
X_train = []
y_train = []
X_test = []
y_test = []

# RNN needs a sequential data structure with num_steps which gives a single output
for i in range(num_steps, train_data_size):
    X_train.append(scaled_training_data[i-num_steps:i, 0])
    y_train.append(scaled_training_data[i, 0])
    
# Create sequential test data
for i in range(num_steps, num_steps+test_data_size):
    X_test.append(scaled_testing_data[i-num_steps:i, 0])
    y_test.append(scaled_testing_data[i, 0])
    
# convert to numpy array and reshape
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



# Build RNN
predictor = Sequential()
# First Layer
predictor.add(LSTM(units=unit_size, return_sequences=True, 
                   input_shape = (X_train.shape[1], 1)))
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
predictor.add(Dense(units=1))
# compile the RNN
predictor.compile(optimizer='adam', loss='mean_squared_error')

# pass training set to RNN
predictor.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batch_size)

# make predictions
predicted_stock_price = predictor.predict(X_test)
#predicted_price = np.append(np.array(scaled_training_data), np.array(predicted_stock_price))
#predicted_price = predicted_price.reshape(-1, 1)
y_test = featureScaler.inverse_transform(np.array(y_test).reshape(-1, 1))
y_pred = featureScaler.inverse_transform(predicted_stock_price)

# Visualize the results
plt.plot(y_pred, color = 'red', label='Predicted Price')
plt.plot(y_test, color = 'blue', label='Actual Price')
plt.title('Prediction using RNN')
plt.xlabel('Date')
plt.ylabel('Stock opening price')
plt.legend()
plt.show()


