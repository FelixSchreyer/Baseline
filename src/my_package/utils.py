from random import randint
import pandas as pd
import dask.dataframe as dd
import requests, zipfile
from matplotlib import pyplot as plt
from tsfresh.utilities.dataframe_functions import roll_time_series
from sklearn.preprocessing import MinMaxScaler
from my_package.add_columns import create_id_column, create_time_column, create_rul_columns

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import os


def download_data():
    if 'train_FD004.txt' not in os.listdir('../../data'):
        print('Downloading Data...')
        # Download the data
        r = requests.get("https://ti.arc.nasa.gov/c/6/", stream=True)
        z = zipfile.ZipFile(StringIO.StringIO(r.content))
        z.extractall('data')
    else:
        print('Using previously downloaded data')


def load_data(data_path):
    operational_settings = ['operational_setting_{}'.format(i + 1) for i in range(3)]
    sensor_columns = ['sensor_measurement_{}'.format(i + 1) for i in range(26)]
    cols = ['engine_no', 'time_in_cycles'] + operational_settings + sensor_columns
    data = pd.read_csv(data_path, sep=' ', header=None, names=cols)
    data = data.drop(cols[-5:], axis=1)
    data['index'] = data.index
    data.index = data['index']
    data['time'] = pd.date_range('1/1/2000', periods=data.shape[0], freq='600s')
    print('Loaded data with:\n{} Recordings\n{} Engines'.format(
        data.shape[0], len(data['engine_no'].unique())))
    print('21 Sensor Measurements\n3 Operational Settings')
    return data


def new_labels(data, labels):
    ct_ids = []
    ct_times = []
    ct_labels = []
    data = data.copy()
    data['RUL'] = labels
    gb = data.groupby(['engine_no'])
    for engine_no_df in gb:
        instances = engine_no_df[1].shape[0]
        r = randint(5, instances - 1)
        ct_ids.append(engine_no_df[1].iloc[r, :]['engine_no'])
        ct_times.append(engine_no_df[1].iloc[r, :]['time'])
        ct_labels.append(engine_no_df[1].iloc[r, :]['RUL'])
    ct = pd.DataFrame({'engine_no': ct_ids,
                       'cutoff_time': ct_times,
                       'RUL': ct_labels})
    ct = ct[['engine_no', 'cutoff_time', 'RUL']]
    ct.index = ct['engine_no']
    ct.index = ct.index.rename('index')
    return ct


def make_cutoff_times(data):
    gb = data.groupby(['engine_no'])
    labels = []

    for engine_no_df in gb:
        instances = engine_no_df[1].shape[0]
        label = [instances - i - 1 for i in range(instances)]
        labels += label

    return new_labels(data, labels)


def feature_importances(X, reg, feats=5):
    feature_imps = [(imp, X.columns[i])
                    for i, imp in enumerate(reg.feature_importances_)]
    feature_imps.sort()
    feature_imps.reverse()
    for i, f in enumerate(feature_imps[0:feats]):
        print('{}: {} [{:.3f}]'.format(i + 1, f[1], f[0]))
    print('-----\n')
    return [f[1] for f in feature_imps[:feats]]


def create_df(directory, files):
    dataframes = []
    var_id = 0
    for folder in files:
        var_id = var_id + 1
        csv_directory = directory + "" + folder
        for filename in os.listdir(csv_directory):
            if filename.endswith('.csv') and filename.startswith("acc"):
                file_path = os.path.join(csv_directory, filename)
            
                df = pd.read_csv(file_path, header=None, sep='[;,]', encoding_errors='replace', engine='python')
                df = create_id_column(df, var_id)
                dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    return df


def plot_data(df):
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
    df = create_df(dir_ref, data)
    df = create_time_column(df)
    df = df.sort_values(by='time')
    df = df.reset_index(drop=True)

    df = df.fillna(method='ffill')

    return df


def roll_data(df, min_timeshift, rolling_direction):
    df = roll_time_series(df, column_id='id', column_sort="time",
                          min_timeshift=min_timeshift, rolling_direction=rolling_direction)
    df = df.assign(bearing_id=df['id'].apply(lambda x: x[0]), time_end=df['id'].apply(lambda x: x[1]))
    df = create_rul_columns(df)
    df['y'] = df['y'].fillna(method='ffill')

    rul_dict = df.groupby('id').agg({'RUL_class': 'min', 'RUL': 'min'}).to_dict()

    df['RUL_rolled_class'] = df['id'].apply(lambda x: rul_dict['RUL_class'].get(x))
    df['RUL_rolled'] = df['id'].apply(lambda x: rul_dict['RUL'].get(x))
    #df['RUL_rolled_class'] = df.groupby('id')['RUL_class'].transform('min')
    #df['RUL_rolled'] = df.groupby('id')['RUL'].transform('min')
    df = df.drop(columns = ['RUL', 'RUL_class'])

    return df


def norm(df):
    scaler = MinMaxScaler()
    def normalize_group(group):
        group['x'] = scaler.fit_transform(group[['x']])
        group['y'] = scaler.fit_transform(group[['y']])
        return group
    normalized_df = df.groupby('id', as_index=False).apply(normalize_group)

    return normalized_df
