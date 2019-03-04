#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:13:20 2019

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
num_epochs = 1
num_batch_size = 32
features = ['open', 'close', 'high', 'low', 'volume']

test_data_size = 60
train_data_size = dailyHistoricalData.shape[0]-test_data_size
testing_data = dailyHistoricalData.tail(2*num_steps)
testing_data_known = dailyHistoricalData.tail(3*num_steps).head(2*num_steps)
training_data = dailyHistoricalData.head(train_data_size)

stock_open_data = dailyHistoricalData.iloc[:, 1:2].values
training_data_open = training_data.iloc[:, 1:2].values
testing_data_open = testing_data.iloc[:, 1:2].values
testing_data_open = testing_data_open.reshape(-1, 1)
testing_data_known_open = testing_data_known.iloc[:, 1:2].values
testing_data_known_open = testing_data_known_open.reshape(-1, 1)

X_train_open = []
y_train_open = []
X_test_open = []
y_test_open = []

for i in range(num_steps, train_data_size):
    X_train_open.append(scaled_training_data_open[i-num_steps:i, 0])
    y_train_open.append(scaled_training_data_open[i+num_steps, 0])