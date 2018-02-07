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
stock_ltfh['y'] =stock_ltfh['Close']
stock_ltfh['y-original'] = stock_ltfh['y']
stock_ltfh['y'] = np.log(stock_ltfh['y'])


stock_ltfh_df = stock_ltfh[['ds','y','y-original']]

print(stock_ltfh_df.tail())
model = Prophet()
model.fit(stock_ltfh_df)

future_data = model.make_future_dataframe(periods = 60)


forecast_data = model.predict(future_data)

print("Log Transformed Data")
print(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


forecast_data_orig = forecast_data # make sure we save the original forecast data
forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])

model.plot(forecast_data)

model.plot_components(forecast_data)

print("Original Data")
print(forecast_data_orig[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()) 





"""
#Building models Motherson Sumi using prophet
stock_mothersumi= stock_mothersumi['Date']

stock_mothersumi['y'] =stock_mothersumi['Adj. Close']

#Building models Anantraj using prophet
stock_anantraj['ds'] = stock_anantraj['Date']

stock_anantraj['y'] =stock_anantraj['Adj. Close']

"""