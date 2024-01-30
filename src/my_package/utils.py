import pandas as pd
from tsfresh.utilities.dataframe_functions import roll_time_series
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from my_package.add_columns import create_id_column, create_time_column, create_rul_columns
import matplotlib.pyplot as plt
# For Segmentation
import ruptures as rpt
import matplotlib.pyplot as plt

import os


def create_df(directory, files):
    '''
    Creates a single dataframe of all csv files in the provided directory.
    Filters for those files starting with 'acc', so those containing accelerator information.

    :param directory: Path to directory of dataset
    :type directory: str
    :param files: list containing titles of csv-files
    :type files: list
    :return: dataframe of csv files
    :rtype: pandas.DataFrame
    '''

    # list to store dataframes of each csv
    dataframes = []
    # helper variable to assign respective experiment-id to rows of dataframe
    var_id = 0
    # iterate through all folders in directory
    for folder in files:
        # update id when changing folder (aka experiment)
        var_id = var_id + 1
        csv_directory = directory + "" + folder
        # iterate through all files in folder
        for filename in os.listdir(csv_directory):
            # only csv files that contain acceleration data of relevance
            if filename.endswith('.csv') and filename.startswith("acc"):
                file_path = os.path.join(csv_directory, filename)
                # seperators can be both ',' and ';'
                df = pd.read_csv(file_path, header=None, sep='[;,]', encoding_errors='strict', engine='python', skiprows=lambda x: x % 2 != 0)
                # Add id as column
                df = create_id_column(df, var_id)
                dataframes.append(df)
    # Create one dataframe of all small ones
    df = pd.concat(dataframes, ignore_index=True)
    return df


def plot_data(df):
    '''
    Prints separate plots for each id in dataframe.
    :param df:
    :type df: pandas.DataFrame
    :return:
    '''

    groups = df.groupby('id')
    for group_id, group_data in groups:
        plt.figure(figsize=(8, 5))
        plt.plot(group_data['time'], group_data['x'])
        plt.plot(group_data['time'], group_data['y'])
        plt.title(f'Time Series for Id {group_id}')
        plt.xlabel('Time')
        plt.ylabel('X/Y')
        plt.show()


def return_df(dir_ref, data):
    '''
    Takes path to folder containing dataset data and list containing names of csv-files.
    Creates Dataframe of all files and subsequently generates a dedicated 'time'-column.

    :param dir_ref: Path to Folder containing Dataset
    :type dir_ref: str

    :param data: Should be a list of valid strings of csv-file names
    :type data: list
    :return: Dataframe containing all file data
    :rtype: pandas.DataFrame
    '''

    df = create_df(dir_ref, data)
    df = create_time_column(df)


    df = df.sort_values(by='time')
    df = df.reset_index(drop=True)

    # Deal with NA/NAN values
    df = df.fillna(method='ffill')

    return df


def roll_data(df, min_timeshift, rolling_direction):
    '''
    Function firstly applies tsfresh's roll_time_series to roll the time series.
    Subsequently, RUL and RUL_Class columns are created.

    :param df: pd.dataframe of arbitrary shape
    :type df: pandas.DataFrame

    :param min_timeshift: int for tsfresh.roll_time_series()
    :type min_timeshift: int

    :param rolling_direction: int for tsfresh.roll_time_series()
    :type rolling_direction: int

    :return: pd.dataframe rolled with RUL columns
    :rtype : pandas.DataFrame
    '''

    # Roll Dataframe
    df = roll_time_series(df, column_id='id', column_sort="time",
                          min_timeshift=min_timeshift, rolling_direction=rolling_direction)


    # Split new id column into an id column containing bearing id and time_end column containing the
    # endpoint of the rolled sequence
    df = df.assign(bearing_id=df['id'].apply(lambda x: x[0]), time_end=df['id'].apply(lambda x: x[1]))

    # Create RUL columns
    df = create_rul_columns(df)

    # Fill empty entries in sensor data
    # df['y'] = df['y'].fillna(method='ffill')
    # df['x'] = df['x'].fillna(method='ffill')


    # Identify RUL and RUL_Class min values for each id (aka rolled sequence)
    rul_dict = df.groupby('id').agg({'RUL_class': 'min', 'RUL': 'min'}).to_dict()

    # Add Columns that contain RUL values for end of rolled sequence
    # these are the relevant values for prediction
    df['RUL_rolled_class'] = df['id'].apply(lambda x: rul_dict['RUL_class'].get(x))
    df['RUL_rolled'] = df['id'].apply(lambda x: rul_dict['RUL'].get(x))

    # df['RUL_rolled_class'] = df.groupby('id')['RUL_class'].transform('min')
    # df['RUL_rolled'] = df.groupby('id')['RUL'].transform('min')

    # Drop columns that are not needed anymore
    df = df.drop(columns=['RUL', 'RUL_class'])

    return df


def norm(df):

    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()

    def normalize_group(group):
        '''
        Takes dataframe and applies scikit-learn scaler to it
        '''
        group[['x', 'y']] = robust_scaler.fit_transform(group[['x', 'y']])
        # group[['x', 'y']] = minmax_scaler.fit_transform(group[['x', 'y']])

        return group

    # Scale df entries separately by id
    df = df.groupby('id', as_index=False).apply(normalize_group)


    return df


def segment(dfs):
    change_points_per_dataframe = []
    for df in dfs:
        grouped = df.groupby('id')

        for id_value, group_df in grouped:
            group_df.reset_index(drop=True)
            model = rpt.Window(width=15360, model="l2", jump=1560, min_size=153600).fit(group_df[['x', 'y']].values)
            # Get the change point indexes
            change_points = model.predict(n_bkps=10)  # Adjust the 'pen' parameter as needed
            print("Change points:", change_points)
            # Convert change point index to [0,1]
            max_values = group_df.idxmax()
            min_values = group_df.idxmin()

            # Calculate time differences
            diff = max_values[1] - min_values

            result = [value / diff for value in change_points]
            print("Change points:", result)

            change_points_per_dataframe.append(result)

    return change_points_per_dataframe

