#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 08:16:54 2019

@author: hrishekesh.shinde
"""

from constraint import *
import pickle
file_open = open('open_prediction', 'rb')
y_pred_open = pickle.load(file_open)
file_open.close()
file_volume = open('volume_prediction', 'rb')
y_pred_volume = pickle.load(file_volume)
file_volume.close()
file_close = open('close_prediction', 'rb')
y_pred_close = pickle.load(file_close)
file_close.close()
file_high = open('high_prediction', 'rb')
y_pred_high = pickle.load(file_high)
file_high.close()
file_low = open('low_prediction', 'rb')
y_pred_low = pickle.load(file_low)
file_low.close()

open_day_dict = {}
available_days = []
stock_start_val = y_pred_open[0][0]
min_gapup = 0.5
stock_start_val = 0

stockPrediction = Problem()

for index in range (0, 60):
    # total difference in a day
    open_day_dict.update({y_pred_open[index][0] : index+1})
    available_days.append(index+1)
    
stockPrediction.addVariable('day', available_days)
stockPrediction.addVariable('action', ['Sell', 'Buy'])


def stockPredictionConstraint(action, day):
    # Buy call in case the share reaches lowest point in a span of 5 days
    if action == 'Buy' and day > 2 and day < 57:
        if y_pred_open[day-1][0] < y_pred_open[day-2][0] and \
        y_pred_open[day-1][0] < y_pred_open[day][0] and \
        y_pred_open[day-1][0] < y_pred_open[day-3][0] and \
        y_pred_open[day-1][0] < y_pred_open[day+1][0]:
            return True
        else:
            return False
    # Sell call in case the share reaches highest point in a span of 5 days    
    if action == 'Sell' and day > 3 and day < 59:
        if y_pred_open[day][0] < y_pred_open[day-1][0] and \
        y_pred_open[day+1][0] < y_pred_open[day-1][0] and \
        y_pred_open[day-2][0] < y_pred_open[day-1][0] and \
        y_pred_open[day-3][0] < y_pred_open[day-1][0]:
            return True
        else:
            return False
    
stockPrediction.addConstraint(stockPredictionConstraint, ['action', 'day'])

calls = stockPrediction.getSolutions()
calls = sorted(calls, key = lambda k: k['day'])

index = 1
# Ensure that a buy call is followed by a sell call
for action_day_dict in calls:
    if (index % 2 == 0 and action_day_dict.get('action') == 'Buy') or \
    (index % 2 == 1 and action_day_dict.get('action') == 'Sell'):
        calls.remove(action_day_dict)
    index += 1

short_term_profit = 0
for call in calls:
    day = call.get('day')
    if call.get('action') == 'Buy':
       short_term_profit -= y_pred_open[day-1][0]
    elif call.get('action') == 'Sell':
       short_term_profit += y_pred_open[day-1][0]
    print('{:5s} on day {} at {:.2f}'.format(call.get('action'), 
          day, y_pred_open[day-1][0]))
print('This agent will help you earn a profit of ${:.2f} per share'.format(short_term_profit))
       
    