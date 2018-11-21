"""
Methods dealing with displaying data
"""

import pandas as pd

def df_all(dataframe):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(dataframe)