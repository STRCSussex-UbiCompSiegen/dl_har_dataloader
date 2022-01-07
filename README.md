# DL-HAR - Dataloader Submodule

This is the dataloader submodule repository of the [dl_har_public repository](https://github.com/STRCSussex-UbiCompSiegen/dl_har_public).

## Contributing to this repository

If you want to contribute to this repository **make sure to fork and clone the main repository [dl_har_public repository](https://github.com/STRCSussex-UbiCompSiegen/dl_har_public) with all its submodules**. To do so please run:

```
git clone --recurse-submodules -j8 git@github.com:STRCSussex-UbiCompSiegen/dl_har_public.git
```
If you want to have your modification be merged into the repository, please issue a **pull request**. If you don't know how to do so, please check out [this guide](https://jarv.is/notes/how-to-pull-request-fork-github/).

## Supported datasets

We currently support the following datasets:

- Opportunity dataset (`opportunity_loso`) and challenge split (`opportunity_challenge`) [[1]](#1)
- RealWorld HAR dataset (`rwhar_loso`) [[3]](#3)
- Skoda Mini Checkpoint dataset (`skoda_split`) [[8]](#5)
- Physical Activity Monitoring dataset (`pamap2_loso`) [[6]](#6)
- Heterogeneity Human Activity Recognition dataset (`hhar_loso`) [[2]](#2)
- University of Sussex-Huawei Locomotion dataset (`shl_loso`) [[4]](#4)

Each YAML configuration file of the above mentioned datasets can be found in the `presets` folder. 

In order to use any of the datasets, please run the `preprocessing.py` file passing along the name of the configuration file as the  `-d` argument.

## Adding a new dataset
The YAML configuration files provide an easy way to add a new dataset. In order to add support for a new dataset define a new YAML configuration file and add it to the `presets` folder. 

Once added run the `preprocessing.py` file passing the name of your newly created configuration file via the `-d` argument.

### Configuration file options

Listed below are all of the 
configuration options available when adding a new dataset:

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

## References

<a id="1">[1]</a> 
Daniel Roggen, Alberto Calatroni, Mirco Rossi, Thomas Holleczek, Kilian Förster,Gerhard Tröster, Paul Lukowicz, David Bannach, Gerald Pirkl, Alois Ferscha, Jakob Doppler, Clemens Holzmann, Marc Kurz, Gerald Holl, Ricardo Chavarriaga, Hesam Sagha, Hamidreza Bayati, Marco Creatura, and José del R. Millàn. 2010. Collecting Complex Activity Datasets in Highly Rich Networked Sensor Environments. https://doi.org/10.1109/INSS.2010.5573462

<a id="2">[2]</a> 
Allan Stisen, Henrik Blunck, Sourav Bhattacharya, Thor S. Prentow, Mikkel B.Kjærgaard, Anind Dey, Tobias Sonne, and Mads M. Jensen. 2015. Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition. https://doi.org/10.1145/2809695.2809718

<a id="3">[3]</a> 
Timo Sztyler and Heiner Stuckenschmidt. 2016. On-Body Localization of Wearable Devices: An Investigation of Position-Aware Activity Recognition. https://doi.org/10.1109/PERCOM.2016.7456521

<a id="4">[4]</a> 
Hristijan Gjoreski, Mathias Ciliberto, Lin Wang, Francisco Javier Ordóñez, Sami Mekki, Stefan Valentin, and Daniel Roggen. 2018. The University of Sussex-Huawei Locomotion and Transportation Dataset for Multimodal Analytics with Mobile Devices. https://doi.org/10.1109/ACCESS.2018.2858933

<a id="5">[5]</a> 
Piero Zappi, Thomas Stiefmeier, Elisabetta Farella, Daniel Roggen, Luca Benini, Gerhard Troster. 2007. Activity Recognition From On-Body Sensors by Classifier Fusion: Sensor Scalability and Robustness. https://doi.org/10.1109/ISSNIP.2007.4496857

<a id="6">[6]</a> 
Attila Reiss and Didier Stricker. 2012. Introducing a New Benchmarked Dataset for Activity Monitoring. https://doi.org/10.1109/ISWC.2012.13
