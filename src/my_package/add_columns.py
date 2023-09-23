import datetime
import pandas as pd
import dask.dataframe as dd

def create_time_column(df):
    datetime.time()
    custom_header_names = ['hour', 'minute', 'second', 'millisecond', 'x', 'y', 'id']

    df.columns.name = 'column_names'
    df.columns = custom_header_names
    default_year = 2023
    default_month = 1
    default_day = 1

    df.reset_index(drop=True)

    # Create a new datetime column
    df['time'] =pd.to_datetime({
        'year': default_year,
        'month': default_month,
        'day': default_day,
        'hour': df['hour'],
        'minute': df['minute'],
        'second': df['second'],
        'millisecond': df['millisecond']
    }, errors='raise')
    columns_to_drop = ['hour', 'minute', 'second', 'millisecond']
    df = df.drop(columns=columns_to_drop)

    new_column_order = ['id', 'time', 'x', 'y']
    df_time = df[new_column_order]

    return df_time


def create_rul_columns(df):
    '''
    Takes a Dataframe and returns it with additional RUL columns.

    Creates column that contains RUL scaled to [0,1] and a column that contains a RUL class.

    :param df: pd.dataframe for which RUL columns shall be created
    :type df: pandas.DataFrame

    :return:
    :rtype: pandas.DataFrame
    '''

    # Get max/min values in time column for each id individually
    max_values = df.groupby('bearing_id')['time'].max()
    min_values = df.groupby('bearing_id')['time'].min()

    # Calculate time differences
    df['date_time_difference'] = max_values[1] - df['time']
    df['max_date_time_difference'] = max_values[1] - min_values[1]

    # Calculate RUL and RUL_class
    # TODO: Decide on RUL Classes
    df['RUL'] = df['date_time_difference'] / df['max_date_time_difference']
    df['RUL_class'] = pd.Series(pd.cut(df['RUL'], bins=[-float('inf'), 0.1, 0.3, float('inf')],
                                     labels=[3, 2, 1], right=False))

    # Drop columns that are no more needed
    df = df.drop(columns=['date_time_difference', 'max_date_time_difference'])

    return df



def create_id_column(df, id_value):
    df['id'] = id_value
    return df
