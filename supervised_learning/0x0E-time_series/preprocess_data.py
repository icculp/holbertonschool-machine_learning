#!/usr/bin/env python3
"""
    Time Series Foreasting
"""
import pandas as pd


def preprocess_data(data_fn, clean_fn):
    ''' preprocess time series csv '''
    # coinbase_fp = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    df = pd.read_csv(data_fn)
    clean = df.dropna(axis=0)
    normal = (clean - clean.mean()) / clean.std()
    normal['Timestamp'] = clean['Timestamp']
    processed = normal[['Timestamp', 'Volume_(Currency)', 'Weighted_Price']]
    processed.to_csv(clean_fn)
    processed.head()
