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
from utils import paint, plot_pie, plot_segment

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
        else:
            self.path_dataset = glob(os.path.join(path_processed, f'{prefix}*.npz'))
        data = [np.load(path, allow_pickle=True) for path in self.path_dataset]
        for i, x in enumerate(data):
            if i == 0:
                self.data = np.c_[np.full(len(x["data"]), i), x["data"]]
                self.target = x["target"]
            else:
                self.data = np.concatenate((self.data, np.c_[np.full(len(x["data"]), i), x["data"]]), axis=0)
                self.target = np.concatenate((self.target, x["target"]))

        self.num_sbj = len(np.unique(self.data[:, 0]))
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

        if self.verbose:
            self.get_info()
            self.get_distribution()

        self.n_channels = self.data.shape[-1] - 1
        self.n_classes = np.unique(self.target).shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        data = torch.FloatTensor(self.data[index])
        target = torch.LongTensor([int(self.target[index])])
        idx = torch.from_numpy(np.array(index))

        return data, target, idx

    def loso_split(self, sbj):
        _data = np.c_[self.data, self.target]

        sbj_data = _data[_data[:, 0] == sbj]
        not_sbj_data = _data[_data[:, 0] != sbj]

        # Normalize data wrt. statistics of the training set
        mean = np.mean(not_sbj_data[:, 1:-1], axis=0)
        std = np.std(not_sbj_data[:, 1:-1], axis=0)

        not_sbj_data[:, 1:-1] = self.normalize(not_sbj_data[:, 1:-1], mean, std)
        sbj_data[:, 1:-1] = self.normalize(sbj_data[:, 1:-1], mean, std)

        return sbj_data, not_sbj_data

    def alter_data(self, data, target, prefix=None):
        self.data = data
        self.target = target
        self.num_sbj = len(np.unique(self.data[:, 0]))
        self.len = data.shape[0]
        if prefix is not None:
            self.prefix = prefix
        return self

    def get_info(self, n_samples=3):
        print(paint(f"[-] Information on {self.dataset} dataset:"))
        print("\t data: ", self.data.shape, self.data.dtype, type(self.data))
        print("\t target: ", self.target.shape, self.target.dtype, type(self.target))

        target_idx = [np.where(self.target == label)[0] for label in set(self.target)]
        target_idx_samples = np.array(
            [np.random.choice(idx, n_samples, replace=False) for idx in target_idx]
        ).flatten()

        for i, random_idx in enumerate(target_idx_samples):
            data, target, index = self.__getitem__(random_idx)
            if i == 0:
                print(paint(f"[-] Information on segment #{random_idx}/{self.len}:"))
                print("\t data: ", data.shape, data.dtype, type(data))
                print("\t target: ", target.shape, target.dtype, type(target))
                print("\t index: ", index, index.shape, index.dtype, type(index))

            path_save = os.path.join(self.path_processed, "segments")

            plot_segment(
                data,
                target,
                index=index,
                prefix=self.name,
                path_save=path_save,
                num_class=len(target_idx),
            )

    def get_distribution(self):
        plot_pie(
            self.target, self.name, os.path.join(self.path_processed, "distribution")
        )

    def normalize(self, data, mean=None, std=None):

        return (data - mean) / std