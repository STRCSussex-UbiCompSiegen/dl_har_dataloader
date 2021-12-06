##################################################
# Class to create a modified dataset object for sensor data also containing meta information.
##################################################
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
##################################################

import os
import numpy as np
import torch
from glob import glob
from torch.utils.data.dataset import Dataset
from dl_har_dataloader.dataloader_utils import paint
from .dataset_utils import sliding_window, normalize

__all__ = ["SensorDataset"]


class SensorDataset(Dataset):
    """
    A dataset class for multi-channel time-series data captured by wearable sensors.
    This class is slightly modified from the original implementation at:
     https://github.com/AdelaideAuto-IDLab/Attend-And-Discriminate
    """

    def __init__(
        self,
        dataset,
        window,
        stride,
        stride_test,
        path_processed,
        name=None,
        prefix=None,
        verbose=False,
    ):
        """
        Initialize instance.
        :param dataset: str. Name of target dataset.
        :param window: int. Sliding window size in samples.
        :param stride: int. Step size of the sliding window for training and validation data.
        :param stride_test: int. Step size of the sliding window for testing data.
        :param path_processed: str. Path to directory containing processed training, validation and test data.
        :param prefix: str. Prefix for the filename of the processed data. Options 'train', 'val', or 'test'.
        :param verbose: bool. Whether to print detailed information about the dataset when initializing.
        :param name: str. What to call this dataset (i.e. train, test, val).
        """

        self.dataset = dataset
        self.window = window
        self.stride = stride
        self.stride_test = stride_test
        self.path_processed = path_processed
        self.verbose = verbose
        self.name = name
        if name is None:
            self.name = 'No name specified'
        if prefix is None:
            self.prefix = 'No prefix specified'
            self.path_dataset = glob(os.path.join(path_processed, '*.npz'))
        elif isinstance(prefix, str):
            self.prefix = prefix
            self.path_dataset = glob(os.path.join(path_processed, f'{prefix}*.npz'))
        elif isinstance(prefix, list):
            self.prefix = prefix
            self.path_dataset = []
            for prefix in prefix:
                self.path_dataset.extend(glob(os.path.join(path_processed, f'{prefix}*.npz')))

        self.data = np.concatenate([np.load(path, allow_pickle=True)['data'] for path in self.path_dataset])
        self.target = np.concatenate([np.load(path, allow_pickle=True)['target'] for path in self.path_dataset])

        self.data = normalize(self.data)

        self.data, self.target = sliding_window(self.data, self.target, self.window, self.stride)

        self.len = self.data.shape[0]
        assert self.data.shape[0] == self.target.shape[0]
        if name is None:
            print(
                paint(
                    f"Creating {self.dataset} HAR dataset of size {self.len} ..."
                )
            )
        else:
            print(
                paint(
                    f"Creating {self.dataset} {self.name} HAR dataset of size {self.len} ..."
                )
            )

        self.n_channels = self.data.shape[-1] - 1
        self.n_classes = np.unique(self.target).shape[0]


    def __len__(self):
        return self.len

    def __getitem__(self, index):

        data = torch.FloatTensor(self.data[index])
        target = torch.LongTensor([int(self.target[index])])
        idx = torch.from_numpy(np.array(index))

        return data, target, idx
