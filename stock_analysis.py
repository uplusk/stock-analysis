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
stock_ltfh_df = quandl.get('NSE/LTFH')
stock_ltfh_df = stock_ltfh_df.reset_index(level=0)

# Motherson Sumi Data
stock_mothersumi_df = quandl.get('NSE/MOTHERSUMI')
stock_mothersumi_df = stock_mothersumi_df.reset_index(level=0)

# Anantraj Ltd. Data
stock_anantraj_df = quandl.get('NSE/ANANTRAJ')
stock_anantraj_df = stock_anantraj_df.reset_index(level=0)

#Building models LTFH using prophet, hence renaming columns
stock_ltfh_df = stock_ltfh_df.rename(columns={'Date':'ds', 'Close':'y'})

stock_mothersumi_df = stock_mothersumi_df.rename(columns={'Date':'ds', 'Close':'y'})

stock_anantraj_df = stock_anantraj_df.rename(columns={'Date':'ds', 'Close':'y'})

# verifying just to make sure
print(stock_anantraj_df.tail())

#historical plot
#stock_ltfh_df.set_index('ds').y.plot()

stock_anantraj_df['y_orig'] = stock_anantraj_df['y']
stock_anantraj_df['y'] = np.log(stock_anantraj_df['y'])

#print(stock_ltfh_df.tail())

# plotting log-transformed data
stock_anantraj_df.set_index('ds').y.plot()

#Running Prophet
model = Prophet()
model.fit(stock_anantraj_df)

future_data = model.make_future_dataframe(periods = 60)


forecast_data = model.predict(future_data)
"""print("Prophet Data")
print(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
"""
model.plot(forecast_data)

model.plot_components(forecast_data)

#Visualizing Prophet models
stock_anantraj_df.set_index('ds', inplace=True)
forecast_data.set_index('ds', inplace=True)

combined_df = stock_anantraj_df.join(forecast_data[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
#print(combined_df.tail())

combined_df['yhat_rescaled'] = np.exp(combined_df['yhat'])
print("Combined df")
print(combined_df.head())

#plotting combined df data
combined_df[['y_orig','yhat_rescaled']].plot()

#to plot a single prediction forecast, we are merging the graphs for a better result
stock_anantraj_df.index = pd.to_datetime(stock_anantraj_df.index) #make sure our index as a datetime object
connect_date = stock_anantraj_df.index[-2] #select the 2nd to last date

mask = (forecast_data.index > connect_date)
prediction_df = forecast_data.loc[mask]

combined_df_final = stock_anantraj_df.join(prediction_df[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
combined_df_final['yhat_rescaled'] = np.exp(combined_df_final['yhat'])
combined_df_final['yhat_upper_rescaled'] = np.exp(combined_df_final['yhat_upper'])
combined_df_final['yhat_lower_rescaled'] = np.exp(combined_df_final['yhat_lower'])

#time to plot
fig, ax1 = plt.subplots()
ax1.plot(combined_df_final.y_orig)
ax1.plot(combined_df_final.yhat_rescaled, color='black', linestyle=':')
ax1.fill_between(combined_df_final.index, np.exp(combined_df_final['yhat_upper']), np.exp(combined_df_final['yhat_lower']), alpha=0.5, color='darkgray')
ax1.set_title('Stock Price(Orange) vs Stock Price Forecast (Black)')
ax1.set_ylabel('Stock Prices (INR)')
ax1.set_xlabel('Date')

L=ax1.legend() #get the legend
L.get_texts()[0].set_text('Actual Price') #change the legend text for 1st plot
L.get_texts()[1].set_text('Forecasted Price') #change the legend text for 2nd plot

print(combined_df.tail(60))
