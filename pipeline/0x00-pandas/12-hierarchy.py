#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

df1 = df1.set_index('Timestamp', drop=True)
df2 = df2.set_index('Timestamp', drop=True)

# print('df1', df2.index.max())
# print(df2.loc[1417411920])

# YOUR CODE HERE

df = pd.concat([df2.loc[1417411980:1417417981],
                df1.loc[1417411980:1417417981]], keys=['bitstamp', 'coinbase'])
df = df.reorder_levels([1, 0], axis=0)
# df = df.reindex(['bitstamp', 'coinbase'], level=0)
# df2 = df.swaplevel(i=- 2, j=- 1, axis=0)
# df = df.reindex(df.index, level=1)
df.sort_index(inplace=True)
# df = df.align(df, level=None, axis=None)

# print(df.loc[('bitstamp', 1417411920)])

print(df)
