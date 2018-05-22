import numpy as np

def train_test_split_time_series(df, test_ratio=0.3, time="Time", labels="Class"):
    df.sort_values(time, inplace=True)
    total_samples = df.shape[0]
    train_idx = int(total_samples * (1 - test_ratio))
    XTrain = df.loc[:train_idx, df.columns != 'Class'].values
    yTrain = df.loc[:train_idx, df.columns == 'Class'].values
    XTest = df.loc[train_idx:, df.columns != 'Class'].values
    yTest = df.loc[train_idx:, df.columns == 'Class'].values
    
    return XTrain, yTrain, XTest, yTest


def reshape_to_batches(matrix, batch_size):
    batch_num = np.ceil(matrix.shape[0] / batch_size)
    modulo = batch_num * batch_size - matrix.shape[0]
    if modulo != 0: 
        padding = np.zeros((int(modulo), matrix.shape[1]))
        matrix = np.vstack((matrix, padding))
        
    return np.array(np.split(matrix, batch_num))


def _3d_to_2d(arr):
    return arr.reshape(arr.shape[0]*arr.shape[1],arr.shape[2])
