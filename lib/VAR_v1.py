# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:54:55 2019

@author: orion.darley
"""
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random
import matplotlib.pyplot as plt

btc = pd.read_csv('C:/Users/orion.darley/Desktop/CS230/btc_apr_1_2012_to_apr_14_2019.csv')
btc = pd.DataFrame(btc)
btc['Date'] = pd.to_datetime(btc['Date'], format ='%M/%d/%Y')
print(btc.head())
print(btc.dtypes)

#btc[['Vol2']] = btc[['Vol2']].apply(pd.to_numeric, downcast = 'float')

"""
Data Prep
"""

"""
EDA
"""

btc.isna().sum() #0

data = pd.DataFrame(btc[['Date','Price', 'Open', 'High', 'Low']])
data = data.sort_values('Date', ascending = True)

# VAR example
# contrived dataset with dependency

"""
fit model - VAR
"""
model = VAR(data[['Price', 'Open', 'High', 'Low']])
model_fit = model.fit()
# make prediction
yhat = model_fit.forecast(model_fit.y, steps=10)
print(yhat)

"""
Plotting
"""

plt.plot(yhat[1])
plt.plot(data['Price'])
