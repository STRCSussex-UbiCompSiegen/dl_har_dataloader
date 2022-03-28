##################################################
# Helper functions for preprocessing.
##################################################
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
##################################################

import numpy as np
import pandas as pd

from scipy.io import loadmat
from io import BytesIO

from scipy.signal import butter, lfilter


def normalize(data, mean, std):
    """Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :param mean: numpy integer array
        Array containing mean values for each sensor channel
    :param std: numpy integer array
        Array containing the standard deviation of each sensor channel
    :return:
        Normalized sensor data
    """
    return (data - mean) / std


def downsample(data_x, data_y, dataset):
    """
    Under construction.
    """

    x_cols = data_x.columns
    data_x = data_x.to_numpy()
    y_name = data_y.name
    data_y = data_y.to_numpy()

    fs = dataset.sr
    factor = dataset.down_sample
    cutoff = fs / (factor * 2)

    init_shapes = (data_x.shape, data_y.shape)

    data_x = butter_lowpass_filter(data_x, cutoff, fs)
    data_x = data_x[::factor]
    data_y = data_y[::factor]

    print(f'Downsampled data from {init_shapes[0]} samples @ {fs}Hz => {data_x.shape} samples @ {fs/factor:.2f}Hz')
    print(f'Downsampled labels from {init_shapes[1]} labels @ {fs}Hz => {data_y.shape} samples @ {fs/factor:.2f}Hz')

    return pd.DataFrame(data_x, columns=x_cols), pd.Series(data_y, name=y_name)


def separate(indices, data_x, data_y, path, prefix):
    for i in range(len(indices)-1):
        start = indices[i]
        stop = indices[i + 1]
        print(f'Separating {prefix} data {start}:{stop} -> {prefix}_data_{i}.npz')
        np.savez_compressed(f'{path}/{prefix}_data_{i}.npz', data=data_x[start:stop], target=data_y[start:stop])


def safe_load(archive, path, dataset):
    if '.mat' in path:
        data = loadmat(BytesIO(archive.read(path)))
        data = data[dataset.variable_name[path]]
        data = pd.DataFrame(data)
    else:
        data = pd.read_csv(BytesIO(archive.read(path)), delimiter=dataset.delimiter, header=dataset.header)

    return data


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y


def handle_missing_data(data):
    data_copy = data.copy()
    for column in data:
        ind = data[column].last_valid_index()
        try:
            data_copy.loc[ind:, column] = data.loc[ind:, column].fillna(0.0)
        except KeyError:
            pass

    # Perform linear interpolation to remove missing values
    data_copy = data_copy.interpolate(method='linear', limit_direction='forward', axis=0)

    # Any remaining missing data are converted to zero
    data_copy = data_copy.fillna(0.0)

    assert not np.isnan(np.sum(data_copy.to_numpy()))

    return data_copy
