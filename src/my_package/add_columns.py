import datetime
import pandas as pd
import dask.dataframe as dd



def create_time_column(df):
    '''
    Takes Dataframe with separate columns for hour, minute, second and millisecond and returns Dataframe with a single
    time column.

    :param df: dataframe for which to calculate time column
    :type df: pandas.DataFrame

    :return: dataframe with dedicated time column
    :rtype: pandas.DataFrame
    '''
    '''
    Takes Dataframe with separate columns for hour, minute, second and millisecond and returns Dataframe with a single
    time column.

    :param df: dataframe for which to calculate time column
    :type df: pandas.DataFrame

    :return: dataframe with dedicated time column
    :rtype: pandas.DataFrame
    '''
    datetime.time()
    custom_header_names = ['hour', 'minute', 'second', 'millisecond', 'x', 'y', 'id']

    df.columns.name = 'column_names'
    df.columns = custom_header_names

    # Necessary for format but can be chosen arbitrarily

    # Necessary for format but can be chosen arbitrarily
    default_year = 2023
    default_month = 1
    default_day = 1

    df.reset_index(drop=True)

    # Create a new datetime column
    df['time'] = pd.to_datetime({
    df['time'] = pd.to_datetime({
        'year': default_year,
        'month': default_month,
        'day': default_day,
        'hour': df['hour'],
        'minute': df['minute'],
        'second': df['second'],
        'millisecond': df['millisecond']
    }, errors='raise')
    # Drop old columns containing information on time
    # Drop old columns containing information on time
    columns_to_drop = ['hour', 'minute', 'second', 'millisecond']
    df = df.drop(columns=columns_to_drop)

    # Change column order
    # Change column order
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
    # Calculate time differences
    df['date_time_difference'] = max_values[1] - df['time']
    df['max_date_time_difference'] = max_values[1] - min_values[1]

    # Calculate RUL and RUL_class
    # TODO: Decide on RUL Classes
    # Calculate RUL and RUL_class
    # TODO: Decide on RUL Classes
    df['RUL'] = df['date_time_difference'] / df['max_date_time_difference']
    df['RUL_class'] = pd.Series(pd.cut(df['RUL'], bins=[-float('inf'), 0.25, 0.5, float('inf')],
                                       labels=[3, 2, 1], right=False))

    # Drop columns that are no more needed
    df = df.drop(columns=['date_time_difference', 'max_date_time_difference'])

    return df


def create_id_column(df, id_value):
    '''

    :param df: input dataframe
    :type df: pandas.DataFrame

    :param id_value: id
    :type id_value: int

    :return: dataframe with additional id column
    :rtype: pandas.DataFrame
    '''

    '''

    :param df: input dataframe
    :type df: pandas.DataFrame

    :param id_value: id
    :type id_value: int

    :return: dataframe with additional id column
    :rtype: pandas.DataFrame
    '''

    df['id'] = id_value
    return df
