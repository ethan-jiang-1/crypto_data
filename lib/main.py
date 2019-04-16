import numpy as np
import pandas as pd
from decimal import Decimal
import pprint
import matplotlib.pyplot as plt
import csv

pd.options.display.float_format = '{:,.2f}'.format

np.random.seed(1)

# btc is Pandas DataFrame of stacked csv columns
btc = pd.read_csv('btc_apr_1_2012_to_apr_14_2019.csv')
print(btc.dtypes)
print(btc['Price'])
btc['Price'] = btc['Price'].apply(lambda p: p.replace(',', ''))
btc['Price'] = btc['Price'].apply(Decimal) # avoid float imprecision
btc['Price'] = btc['Price'].apply(lambda p: round(p,2))
print(btc['Price'])
# changes = btc['Change %']
# ranges = btc['High'] - btc['Low']

# def initialize_parameters():
#
# def initialize_parameters_deep():
#
# def linear_forward():
#
# def linear_activation_forward():
#
# def L_model_forward():
#
# def compute_cost():
#
# def linear_backward():
#
# def linear_activation_backward():
#
# def L_model_backward():
#
# def update_parameters():
#
# def model():
