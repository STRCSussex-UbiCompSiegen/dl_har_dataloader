##################################################
# All functions related to preprocessing the sensor datasets. Run this file in order to obtain the datasets.
##################################################
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
##################################################

import os
import sys
import argparse
from io import BytesIO

import numpy as np
import pandas as pd

if __package__ is None or __package__ == '':
    from preprocessing_utils import separate, downsample, safe_load, handle_missing_data
    from dataloader_utils import makedir, create_rwhar_dataset
else:
    from .preprocessing_utils import separate, downsample, safe_load, handle_missing_data
    from .dataloader_utils import makedir, create_rwhar_dataset

# needed for relative imports to work
p = os.path.abspath('../..')
if p not in sys.path:
    sys.path.append(p)


def get_labels_from_file(labels_path, zf, user, dataset):
    labels = pd.read_csv(BytesIO(zf.read(labels_path)), delimiter=dataset.delimiter, header=dataset.header)  # Note
    # this assumes that the labels are in the same archive as the corresponding data.

    labels = labels[dataset.label_column]

    if user in ['train', 'test', 'val']:
        labels = labels[dataset.split[user][0] * len(labels):dataset.split[user][1] * len(labels)]

    return labels


def preprocess(data, labels_path, zf, user, dataset):
    # if dataset.exclude_channels is not None:
    #     data = data.drop(labels=dataset.exclude_channels, axis=1)

    if dataset.exclude_labels is not None:
        data = data[~data[dataset.label_column].isin(dataset.exclude_labels)]

    data_x = data[dataset.sensor_columns]

    if dataset.label_files is not None:
        data_y = get_labels_from_file(labels_path, zf, user, dataset)
    else:
        data_y = data[dataset.label_column]

    if dataset.label_map is not None:
        data_y = data_y.replace(dataset.label_map)
        data_y = data_y.fillna(0)

    data_x = handle_missing_data(data_x)

    if dataset.down_sample:
        data_x, data_y = downsample(data_x, data_y, dataset)

    return data_x, data_y


def separate_user_data(data, user, dataset):
    if dataset.user_map is not None:
        user = list(dataset.user_map.keys())[int(user[-1])]  # User is a string ending in an integer (i.e. User_0)
    return data[data[dataset.user_column] == user]


def load_data(target, zf, user, dataset):
    data = safe_load(zf, target, dataset)

    if dataset.user_column is not None:
        data = separate_user_data(data, user, dataset)

    if user in ['train', 'test', 'val']:
        data = data[int(dataset.split[user][0] * len(data)):int(dataset.split[user][1] * len(data))]

    return data


def load_and_preprocess(target, zf, user, labels_path, dataset):
    if dataset.is_multiple_zips:
        archive = zf[target[0]]
        path = target[1]

    else:
        archive = zf
        path = target

    data = load_data(path, archive, user, dataset)
    x, y = preprocess(data, labels_path, archive, user, dataset)

    return x, y


def safe_hstack(top, bottom):

    if top.shape[0] > bottom.shape[0]:
        bottom = np.concatenate((bottom, np.zeros((top.shape[0] - bottom.shape[0], bottom.shape[1]))), axis=0)
    elif bottom.shape[0] > top.shape[0]:
        top = np.concatenate((top, np.zeros((bottom.shape[0] - top.shape[0], top.shape[1]))), axis=0)

    return np.hstack((top, bottom))


def iter_files(dataset, zf, user):
    if dataset.n_users > 1:
        files = dataset.data_files[user]
    else:
        files = dataset.data_files

    label_files = dataset.label_files

    file_end_indices = [0]

    for i, filepath in enumerate(files):
        if label_files is not None:
            labels_path = label_files[user][i]
            x, y = load_and_preprocess(filepath, zf, user, labels_path, dataset)
        else:
            x, y = load_and_preprocess(filepath, zf, user, None, dataset)
        print(f'... file(s) {filepath} -> User_{user}_data')

        if i == 0:
            data_x = np.array(x)
        else:
            if dataset.n_channels_per_file != dataset.n_channels:
                data_x = safe_hstack(data_x, x)
                assert len(data_x) == len(x), "All sensor channels must contain the same amount of samples."
            else:
                data_x = np.vstack((data_x, x))

        if dataset.multiple_recordings_per_user:
            if i == 0:
                data_y = np.array(y, dtype=np.uint8)
            else:
                data_y = np.concatenate((data_y, y))
            file_end_indices.append(len(data_y))

    if not dataset.multiple_recordings_per_user:
        data_y = np.array(y, dtype=np.uint8)
        file_end_indices.append(len(data_y))

    return data_x, data_y, file_end_indices


def preprocess_dataset(dataset, args):
    zf = dataset.open_zip()

    makedir(f'{args.output_dir}/{dataset.name}')

    # Special cases
    if dataset.name == 'rwhar':

        data = create_rwhar_dataset(dataset.data_dir)
        data['activity'] = data['activity'].map(dataset.label_map)

        for user in range(15):
            x, y = preprocess(data[data['subject'] == user], None, zf, user, dataset)
            data_x = np.array(x)
            data_y = np.array(y, dtype=np.uint8)
            print(
                f'Saving file User_{str(user).zfill(3)}_data.npz containing data {data_x.shape}, labels {data_y.shape}')
            np.savez_compressed(f'{args.output_dir}/{dataset.name}/User_{str(user).zfill(3)}.npz', data=data_x,
                                target=data_y)

    else:

        if dataset.is_multiple_files:

            # Multiple users case
            if dataset.n_users > 1:

                for user in dataset.data_files.keys():

                    data_x, data_y, file_end_indices = iter_files(dataset, zf, user)

                    if args.separate:
                        separate(file_end_indices, data_x, data_y, f'{args.output_dir}/{dataset.name}', user)
                    else:
                        print(
                            f'Saving file {user}.npz containing data {data_x.shape}, labels {data_y.shape}')
                        np.savez_compressed(f'{args.output_dir}/{dataset.name}/{user}.npz', data=data_x,
                                            target=data_y)
            # Single user case
            else:

                for split in ['train', 'test', 'val']:

                    data_x, data_y, _ = iter_files(dataset, zf, split)

                    print(
                        f'Saving file {split}.npz containing data {data_x.shape}, labels {data_y.shape}')
                    np.savez_compressed(f'{args.output_dir}/{dataset.name}/{split}.npz', data=data_x,
                                        target=data_y)

        else:
            print('Datasets contained in a single file not yet implemented')


def get_args():
    parser = argparse.ArgumentParser(
        description='Preprocess OPPORTUNITY dataset')

    parser.add_argument(
        '-d', '--dataset', type=str, help='Target dataset', required=True)
    parser.add_argument(
        '-s', '--separate', type=bool, help='Keep files separate (for CausalBatch). Defaults to False.', default=False,
        required=False)
    parser.add_argument(
        '-i', '--input_dir', type=str, help='Directory containing the raw data (zip file). Defaults to ../data/raw',
        default='../data/raw', required=False)
    parser.add_argument(
        '-o', '--output_dir', type=str, help='Directory to contain the processed data. Defaults to ../data/',
        default='../data', required=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    if __package__ is None or __package__ == '':
        from dataloader import DatasetLoader, load_preset
    else:
        from .dataloader import DatasetLoader, load_preset

    args = get_args()

    config = load_preset(args.dataset)
    loader = DatasetLoader(data_dir=args.input_dir, **config)  # Get config from data_config and unpack

    preprocess_dataset(loader, args)
