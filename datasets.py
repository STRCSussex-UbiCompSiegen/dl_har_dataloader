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
from torch.utils.data import Sampler
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
            causal=False
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

        self.data = np.array([np.load(path, allow_pickle=True)['data'] for path in self.path_dataset])
        self.target = np.array([np.load(path, allow_pickle=True)['target'] for path in self.path_dataset])
        for i in range(len(self.data)):
            self.data[i] = normalize(self.data[i])
            self.data[i], self.target[i] = sliding_window(self.data[i], self.target[i], self.window, self.stride)

        if causal:
            self.lengths = [len(array) for array in self.target]

        self.data = np.concatenate(self.data)
        self.target = np.concatenate(self.target)
        self.len = self.data.shape[0]
        self.n_classes = np.unique(self.target).shape[0]

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


    def __len__(self):
        return self.len

    def __getitem__(self, index):

        data = torch.FloatTensor(self.data[index])
        target = torch.LongTensor([int(self.target[index])])
        idx = torch.from_numpy(np.array(index))

        return data, target, idx


class CausalSampler(Sampler):
    """Sampler class to implement causal batch sampling for recurrent and LSTM neural networks."""

    def __init__(self, dataset, batchsize, batchlen, num_batches, shuffle, drop_last):

        self.dataset = dataset
        self.batchsize = batchsize
        self.batchlen = batchlen
        self.num_batches = num_batches
        self.shuffle = shuffle
        self.drop_last = drop_last

        window_size = dataset.window
        assert (
                    window_size / dataset.stride).is_integer(), 'in order to generate sequential batches, the sliding window length ' \
                                                                'must be divisible by the step.'

        starts = [[x for x in range(0, i - int(((batchlen * window_size) + 1) / dataset.stride))]
                  for i in dataset.lengths]

        print(dataset.lengths)

        for i in range(1, len(starts)):
            starts[i] = [x + 1 + starts[i - 1][-1]
                         + int(((batchlen * window_size) + 1) / dataset.stride) for x in starts[i]]

        self.starts = [val for sublist in starts for val in sublist]

        self.step = lambda x: [int(x + i * window_size / dataset.stride) for i in range(batchlen)]

        if shuffle:
            np.random.shuffle(self.starts)

        self.batches = np.empty((batchsize, batchlen), dtype=np.int32)

        if self.num_batches != -1:
            self.num_batches = int(self.num_batches * self.batchsize)  # Convert num_batches to number of metabatches.
            if num_batches > len(starts):
                self.num_batches = -1

    def __iter__(self):

        for i, start in enumerate(self.starts[0:self.num_batches]):

            batch = np.array([i for i in self.step(start)], dtype=np.int32)

            self.batches[i % self.batchsize] = batch

            if i % self.batchsize == self.batchsize - 1:
                self.batches = self.batches.transpose()
                for pos, batch in enumerate(self.batches):

                    yield batch
                    self.batches = np.empty((self.batchsize, self.batchlen), dtype=np.int32)

            if self.drop_last is False and i == len(self.starts) and i % self.batchsize != 0:

                self.batches = self.batches[0:i % self.batchsize]
                self.batches = self.batches.transpose()
                for pos, batch in enumerate(self.batches):
                    yield batch
