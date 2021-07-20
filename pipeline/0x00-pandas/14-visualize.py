#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df.drop(columns='Weighted_Price', inplace=True)
df.rename(columns={"Timestamp": "Date"}, inplace=True)
# df['Date'] = pd.DatetimeIndex(df['Date'])
df['Date'] = pd.to_datetime(df['Date'], unit='ms')
df.set_index('Date', inplace=True, drop=False)
# print(df)
# Dates = pd.DatetimeIndex(df['Date'])
df['Close'].fillna(method='ffill', inplace=True)
# print(any(df['Close'].isna()))
df['High'].fillna(df.Close, inplace=True)
df['Low'].fillna(df.Close, inplace=True)
df['Open'].fillna(df.Close, inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)
df = df[df.index.year >= 2017]
# df = df.set_index('Date', drop=True)
# df.set_index('Date', inplace=True)
# groups = df.groupby([Dates.day])
High = df.groupby(pd.Grouper(key='Date', freq='1D')).max()['High']
Low = df.groupby(pd.Grouper(key='Date', freq='1D')).min()['Low']
Open = df.groupby(pd.Grouper(key='Date', freq='1D')).mean()['Open']
Close = df.groupby(pd.Grouper(key='Date', freq='1D')).mean()['Close']
VolumeBTC = df.groupby(pd.Grouper(key='Date', freq='1D')).sum()['Volume_(BTC)']
VolumeCurrency = df.resample('1D').sum()['Volume_(Currency)']
# df.groupby(pd.Grouper(key='Date', freq='1D')).sum()['Volume_(Currency)']

# print('vol', VolumeCurrency)

df = pd.concat([Open, High, Low, Close, VolumeBTC, VolumeCurrency], axis=1)
# Open = groups.max()
# print(Open)
'''df['High'] = df['High'].resample('1D').max()
df['Low'] = df['High'].resample('1D').min()
# def custom_resampler_mean(arraylike):
#    """ takes mean of sample """
#    return np.sum(arraylike) / len(arraylike)
df['New_Open'] = df['Open'].resample('1D')#.mean()
df['Close'] = df['Close'].resample('1D').mean()
df['Volume_(BTC)'] = df['Volume_(BTC)'].resample('1D').sum()
df['Volume_(Currency)'] = df['Volume_(Currency)'].resample('1D').sum()'''
# df_counts = df.your_date_column.resample('D')
# pd.to_datetime(df['Datetime'], unit='s')
d = df.plot(kind='line')
d.set_xticks([x.strftime("%Y - %m") for x in df.index])
d.set_yticks([0, int(df['Volume_(Currency)'].max())])
plt.show()
# print(df.describe())
# YOUR CODE HERE
