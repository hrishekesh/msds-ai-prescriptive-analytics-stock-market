#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 20:28:51 2019

@author: hrishekesh.shinde
"""
from constraint import *
import pickle
# read the predicted data from pickle files
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

day_diff = []
max_diff_high = []
min_diff_high = []

day_diff_dict = {}
max_diff_high_dict = {}
min_diff_high_dict = {}

stockCallProblem_daydiff = Problem()
stockCallProblem_maxdiff = Problem()
stockCallProblem_mindiff = Problem()
for index in range (0, 60):
    # total difference in a day
    current_day_diff = y_pred_close[index][0] - y_pred_open[index][0]
    day_diff_dict.update({current_day_diff : index+1})
    day_diff.append(current_day_diff)
    # max increase in a day
    current_max_diff_high = y_pred_high[index][0] - y_pred_open[index][0]
    max_diff_high_dict.update({current_max_diff_high : index+1})
    max_diff_high.append(current_max_diff_high)
    # min decrease in a day
    current_min_diff_high = y_pred_low[index][0] - y_pred_open[index][0]
    min_diff_high_dict.update({current_min_diff_high : index+1})
    min_diff_high.append(current_min_diff_high)
    
# calculate predicted differences in a day

stockCallProblem_daydiff.addVariable('dayDiff', day_diff)
stockCallProblem_maxdiff.addVariable('maxDiff', max_diff_high)
stockCallProblem_mindiff.addVariable('minDiff', min_diff_high)

stockCallProblem_daydiff.addVariable('action', ['Short Sell', 'Intraday Buy', 'No Action'])
stockCallProblem_maxdiff.addVariable('action', ['Short Sell', 'Intraday Buy', 'No Action'])
stockCallProblem_mindiff.addVariable('action', ['Short Sell', 'Intraday Buy', 'No Action'])

# constraint for difference between close price and open price - for intraday buy
def addProfitConstraint_daydiff(dayDiff, action):
    minDiffBuy = 0.16
    if action == 'Intraday Buy' and dayDiff > minDiffBuy:
        return True
    return False
# constraint for difference between day high price and open price - for intraday buy
def addProfitConstraint_maxdiff(maxDiff, action):
    minDiffBuy = 0.16
    if action == 'Intraday Buy' and maxDiff > minDiffBuy:
        return True
    return False

# constraint for difference between open price and day low price - for short sell
def addProfitConstraint_mindiff(minDiff, action):
    minDiffBuy = -0.16
    if action == 'Short Sell' and minDiff < minDiffBuy:
        return True


stockCallProblem_daydiff.addConstraint(addProfitConstraint_daydiff, ['dayDiff', 'action'])
stockCallProblem_maxdiff.addConstraint(addProfitConstraint_maxdiff, ['maxDiff', 'action'])
stockCallProblem_mindiff.addConstraint(addProfitConstraint_mindiff, ['minDiff', 'action'])

calls_daydiff = stockCallProblem_daydiff.getSolutions()
calls_maxdiff = stockCallProblem_maxdiff.getSolutions()
calls_mindiff = stockCallProblem_mindiff.getSolutions()

calls = []
calls.extend(calls_daydiff)
calls.extend(calls_maxdiff)
calls.extend(calls_mindiff)
calls.reverse()

intraday_profit = 0
print('The recommendations for intraday trades are as below:')
for call in calls:
    day = 0
    start_price = 0
    target_price = 0
    if call.get('dayDiff') is not None:
        day = day_diff_dict.get(call.get('dayDiff'))
        start_price = y_pred_open[day-1][0]
        target_price = y_pred_close[day-1][0]
        intraday_profit += (target_price - start_price)
    elif call.get('maxDiff') is not None:
        day = max_diff_high_dict.get(call.get('maxDiff'))
        start_price = y_pred_open[day-1][0]
        target_price = y_pred_high[day-1][0]
        intraday_profit += (target_price - start_price)
    elif call.get('minDiff') is not None:
        day = min_diff_high_dict.get(call.get('minDiff'))
        start_price = y_pred_open[day-1][0]
        target_price = y_pred_low[day-1][0]
        intraday_profit += (start_price - target_price)
    print('{:15s} on day {} at {:.2f} with target {:.2f}'.format(call.get('action'), 
          day, start_price, target_price))
print('This agent will help you earn a profit of ${:.2f} per share'.format(intraday_profit))
print('These results are with the assumption that first action if any is taken at market opening')