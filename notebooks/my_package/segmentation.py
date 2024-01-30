import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ruptures as rpt

def segment(df_list):
    normalized_dfs = []
    for df in df_list:
        df['norm_axis'] = (normalize(df))
        normalized_dfs.append(df)

    merged_df = normalized_dfs[0]
    for normalized_df in normalized_dfs[1:]:
        merged_df = pd.concat([normalized_df, merged_df], ignore_index=True)

    # NOW: merged datasets on same frame, time for segmenation

    data = np.array([merged_df['x'], merged_df['y']]).T
    model = "normal"
    algo = rpt.Window(model=model, width=5000).fit(data)

    result = algo.predict(n_bkps=3) #Breakpoints
    cats = [1, 2, 3, 4]

    # Visualize the change points
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['norm_axis'], merged_df['x'], label='Value DF1')
    plt.plot(merged_df['norm_axis'], merged_df['y'], label='Value DF2')
    for cp in result:
        plt.axvline(x=(cp/result[-1]), color='r', linestyle='--', label="Change Point")
    plt.xlabel("Normalized Time")
    plt.ylabel("Value")
    plt.title("Change Point Detection for Value DF1")
    plt.legend()
    plt.grid(True)
    plt.show()

    df_to_return = []

    for df in df_list:
        df['cats'] = None
        for threshold, value in zip(result, cats):
            df.loc[df['RUL'] < threshold, 'integer_column'] = value

        df_to_return.append(df)


def normalize(df):
    max_time = (df['time'].max())
    min_time = (df['time'].min())

    return (df['time'] - min_time) / (max_time - min_time)