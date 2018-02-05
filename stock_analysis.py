# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
# Quandl for financial analysis, pandas and numpy for data manipulation

# fbprophet for additive models, #pytrends for Google trend data

import quandl

import pandas as pd

import numpy as np

from fbprophet import Prophet

# matplotlib pyplot for plotting
import matplotlib.pyplot as plt
import matplotlib

# stocker library
from stocker import Stocker

# set api key
quandl.ApiConfig.api_key = "sY8rR9LTz4kT5z2cs94v"

# L&T Finance Holdings data
stock_ltfh = quandl.get('NSE/LTFH')
stock_ltfh = stock_ltfh.reset_index(level=0)

# Motherson Sumi Data
stock_mothersumi = quandl.get('NSE/MOTHERSUMI')
stock_mothersumi = stock_mothersumi.reset_index(level=0)

# Anantraj Ltd. Data
stock_anantraj = quandl.get('NSE/ANANTRAJ')
stock_anantraj = stock_anantraj.reset_index(level=0)

#Building models LTFH using prophet
stock_ltfh['ds'] = stock_ltfh['Date']
stock_ltfh['y'] =stock_ltfh['Adj. Close']
stock_ltfh['y'] = np.log(stock_ltfh['y'])


stock_ltfh_df = stock_ltfh[['ds','y']]

print(stock_ltfh_df.head())
m = Prophet()
m.fit(stock_ltfh)




"""
#Building models Motherson Sumi using prophet
stock_mothersumi= stock_mothersumi['Date']

stock_mothersumi['y'] =stock_mothersumi['Adj. Close']

#Building models Anantraj using prophet
stock_anantraj['ds'] = stock_anantraj['Date']

stock_anantraj['y'] =stock_anantraj['Adj. Close']

"""