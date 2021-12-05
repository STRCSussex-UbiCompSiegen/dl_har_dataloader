##################################################
# Dataloader object for sensor data based on SensorDataset object
##################################################
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
# Author: Marius Bock
# Email: marius.bock@uni-siegen.de
##################################################

import zipfile
import os

from dataloader_utils import makedir
import progressbar
import yaml


class ProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


class DatasetLoader:

    def __init__(self, name, data_dir, data_archive, data_files, class_map, n_classes, n_channels, label_column,
                 sensor_columns, sr, user_column=None, user_map=None, down_sample=False, label_map=None,
                 variable_name=None, url=None, label_files=None, exclude_labels=None, n_channels_per_file=None,
                 header=None, delimiter=None, needs_sync=False, time_column=None):
        """
        :param str name: The name of the dataset, in lower case i.e. 'opportunity'.
        :param str data_dir: The directory containing the raw data (zip file).
        :param str|list (str) data_archive: The name of the zip file i.e. 'OpportunityUCIDataset.zip', or a list of
         names if the dataset is split over multiple archives.
        :param dict (str: list)| dict (str: tuple) data_files: If the dataset is contained in one archive, a dict of the
         form {int: str|list(str),...} where the key is the user number and the value represents the file or files which
         contain the data for that user.
         If the dataset is split over multiple archives, a dict of the form
         {int: tuple(int, str)} where the key indicates the user number, and the value is a tuple where the first entry
         indicates which archive the file is contained in, and the second indicates the path of the file within the
         archive.
         If the dataset is contained in a single file, this should just be a string giving the filepath.
        :param list (str) class_map: The names of the activity classes in the dataset.
        :param int n_classes: The number of classes in the dataset.
        :param int n_channels: The number of sensor channels from the dataset to use.
        :param int label_column: The column containing the activity label.
        :param iterable (int) sensor_columns: The columns containing the sensor data.
        :param int sr: The sample rate of the dataset.
        :param int user_column: The column indicating the user. Only if multiple user data is given in the same file.
        :param bool|int down_sample: Whether to downsample the data, and if so what factor to use. If downsampling is
         required, this should be an integer specifying the downsampling factor.
        :param dict (any: int) label_map: If necessary, a dictionary giving the mapping between existing labels and
        integer labels.
        :param str variable_name: Required if the dataset is contained in a .mat file. Specifies the variable containing
        the dataset.
        :param str|list (str) url: The URL where the dataset can be retrieved from, if not found in data_dir.
        :param dict (str: list)|dict (str: tuple) label_files: The files containing labels, should be a dict of the form
        {'train': list_of_filepaths, 'val': list_of_filepaths, 'test': list_of_filepaths} if the data is contained
        within one archive, or {'train': list_of_tuples, 'val': list_of_tuples, 'test': list_of_tuples} where the first
        entry in  each tuple is the index of the archive containing the file, and the second is the filepath.
        :param int n_channels_per_file: If sensor channels are split across multiple files, indicates how many per file.
        :param bool header: Which line to use as column names when reading data. None = no header, columns are numbered
        :param str delimiter: Character marking the separation between values in the data files.
        Typically ' ', ',' or '|'.
        :param bool needs_sync: Whether the sensor channels need to be synchronized.
        :param str|int time_column: Column containing timestamps for synchronization of data.
        """

        self.__dict__.update(locals())

        if self.delimiter is None:
            self.delimiter = ','

        if n_channels_per_file is None:
            self.n_channels_per_file = n_channels
        else:
            self.n_channels_per_file = n_channels_per_file

        self.is_multiple_files = not isinstance(self.data_files, str)

        self.is_multiple_zips = isinstance(self.data_archive, list)

        self.multiple_recordings_per_user = False

        if self.name in ['pamap2', 'opportunity']:
            self.multiple_recordings_per_user = True

    def open_zip(self):
        """Find the data and return a zipfile or list of zipfiles
        """

        path = self.check_data()

        if isinstance(path, list):
            zf = [zipfile.ZipFile(path) for path in path]
        else:
            zf = zipfile.ZipFile(path)

        return zf

    def check_data(self):
        """Try to access to the file and checks if dataset is in the data directory
           In case the file is not found try to download it from original location
        """

        if not os.path.exists(self.data_dir):
            makedir(self.data_dir)

        if isinstance(self.data_archive, list):
            paths = [f'{self.data_dir}/{file}' for file in self.data_archive]
            print(f'Checking datasets {paths}')
            for i, path in enumerate(paths):
                if not os.path.isfile(path):
                    self.get_data_from_url(path, i)
        else:
            paths = f'{self.data_dir}/{self.data_archive}'
            if not os.path.isfile(paths):
                self.get_data_from_url(paths)

        return paths

    def get_data_from_url(self, path, index=None):
        """ Download the dataset from the given url. If multiple there are multiple urls, requires an index.

        :param str path: The path where the dataset should be put.
        :param int index: Which URL to download from (if self.url is a list).
        :return: None

        """
        import urllib.request
        # When dataset not found, try to download it
        print('... dataset path {0} not found'.format(path))
        print('Attempting to download ...')
        if index is not None:
            origin = self.url[index]
        else:
            origin = self.url
        print('... downloading data from {0}'.format(origin))
        urllib.request.urlretrieve(origin, path, ProgressBar())


def load_preset(preset):
    """Load a preset from a YAML config file in ./presets.

    :param str preset: Name of preset to load.
    :return dict: Dictionary containing arguments for a DatasetLoader instance read from the YAML preset file.
    """

    here = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(here, f'presets/{preset}.yaml')) as f:
        config = yaml.safe_load(f)

    return config
