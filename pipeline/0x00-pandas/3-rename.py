#!/usr/bin/env python3

from typing import get_type_hints
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
# df = df[::-1]
df = df[['Datetime', 'Close']]
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
print(df.tail())
