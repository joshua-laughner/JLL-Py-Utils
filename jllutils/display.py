"""
Methods dealing with displaying data
"""

import pandas as pd


def df_all(dataframe):
    """
    Print all rows of a data frame.

    :param dataframe: the data frame to print.
    :type dataframe: :class:`pandas.DataFrame`

    :return: None
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(dataframe)
