#!/usr/bin/env python3
"""
    Pandas project
"""
import pandas as pd


stupid_dict = {'First': [0.0, 0.5, 1.0, 1.5],
               'Second': ['one', 'two', 'three', 'four']}
rows = [chr(65 + i) for i in range(4)]

df = pd.DataFrame(stupid_dict, index=rows, columns=sorted(stupid_dict.keys()))
