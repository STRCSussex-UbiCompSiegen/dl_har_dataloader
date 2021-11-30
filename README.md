# dl_har_dataloader

## Configuration files

The YAML configuration files provide an easy way to add a new dataset. Listed below are all of the 
configuration options available when adding a new dataset.

- *name*: The name of the dataset, in lower case i.e. 'opportunity'.
- *data_archive*: The name of the zip file i.e. 'OpportunityUCIDataset.zip', or a list of
         names if the dataset is split over multiple archives.
- *data_files*: If the dataset is contained in one archive, a dict of the
         form {*int*: *str|list(str)*} where the key is the user number and the value is a string or list of strings 
         giving the path(s) to the files containing the data for that user.
         If the dataset is split over multiple archives, a dict of the form
         {*int*: *tuple(int, str)*} where the key indicates the user number, and the value is a tuple where the first 
         entry indicates which archive the file is contained in, and the second the path(s) to the files containing the 
         data for that user. If the dataset is contained in a single file, this should just be a string giving the filepath.
- **class_map** *iterable (str)*: The names of the activity classes in the dataset, in order so that the element at class_map[i] gives the
  name of the class i.
- **n_classes** *int*: The number of classes in the dataset.
- **n_channels** *int*: The number of sensor channels from the dataset to use.
- **label_column** *int|str*: The column containing the activity label.
- **sensor_columns** *iterable (int|str)*: The column(s) containing the sensor data.
- **sr** *int*: The sample rate of the dataset.
- **user_column** *int|str*: The column indicating the user, if the data of multiple users are given in the same file.
- **down_sample** *bool*: Whether to downsample the data.
- **down_sample_factor** *int* What factor to downsample by. 
- **label_map** *dict (any:int)*: If necessary, a dictionary giving the mapping between existing labels and
        integer labels.
- **variable_name** *str*: Required if the dataset is contained in a .mat file. Specifies the variable containing
        the dataset.
- **url** *str*: The URL where the dataset can be retrieved from, if not found in data_dir.
- **label_files** *dict (str: list (str))|dict (str: list(tuple(int, str)))*: The files containing labels, should be a dict of the form
        {'train': list_of_filepaths, 'val': list_of_filepaths, 'test': list_of_filepaths} if the data is contained
        within one archive, or {'train': list_of_tuples, 'val': list_of_tuples, 'test': list_of_tuples} where the first
        entry in  each tuple is the index of the archive containing the file, and the second is the filepath.
- **n_channels_per_file** _int_: If sensor channels are split across multiple files, indicates how many per file.
- **header** _int_: Which line to use as column names when reading data. None = no header, columns are numbered.
- **delimiter** *str*: Character marking the separation between values in the data files. Typically ' ', ',' or '|'.