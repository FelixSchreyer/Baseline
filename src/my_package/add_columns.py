import datetime
import pandas as pd

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
    df['time'] = pd.to_datetime({
        'year': default_year,
        'month': default_month,
        'day': default_day,
        'hour': df['hour'],
        'minute': df['minute'],
        'second': df['second'],
        'millisecond': df['millisecond']
    })
    columns_to_drop = ['hour', 'minute', 'second', 'millisecond']
    df = df.drop(columns=columns_to_drop)

    new_column_order = ['id', 'time', 'x', 'y']
    df_time = df[new_column_order]

    return df_time


def create_rul_columns(df):
    df_helper = pd.DataFrame()
    max_values = df.groupby('bearing_id')['time'].max()
    min_values = df.groupby('bearing_id')['time'].min()

    # Calculate time differences using vectorized operations
    df_helper['date_time_difference'] = max_values[1] - df['time']
    df_helper['max_date_time_difference'] = max_values[1] - min_values[1]

    # Calculate the result column using vectorized operations
    df['RUL'] = df_helper['date_time_difference'] / df_helper['max_date_time_difference']

    df['RUL_cat'] = (df['RUL']*5).astype('int')
    return df


def create_id_column(df, id_value):
    df['id'] = id_value
    return df
