##################################################
# Helper funtions for data loading.
##################################################
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
# Author: Marius Bock
# Email: marius.bock@uni-siegen.de
##################################################

import os
import zipfile
from io import BytesIO

import pandas as pd


def create_rwhar_dataset(raw_dir):
    """
    Function to create RWHAR dataset (containing only acceleration data).
    Original Author : Sandeep Ramachandra, sandeep.ramachandra@student.uni-siegen.de
    Modified by: Marius Bock
    :return: pandas dataframe containing all data
    """
    RWHAR_ACTIVITY_NUM = {
        "climbingdown": 'climbing_down',
        "climbingup": 'climbing_up',
        "jumping": 'jumping',
        "lying": 'lying',
        "running": 'running',
        "sitting": 'sitting',
        "standing": 'standing',
        "walking": 'walking',
    }

    RWHAR_BAND_LOCATION = {
        "chest": 1,
        "forearm": 2,
        "head": 3,
        "shin": 4,
        "thigh": 5,
        "upperarm": 6,
        "waist": 7,
    }

    def check_rwhar_zip(path):
        # verify that the path is to the zip containing csv and not another zip of csv

        if any(".zip" in filename for filename in zipfile.ZipFile(path, "r").namelist()):
            # There are multiple zips in some cases
            with zipfile.ZipFile(path, "r") as temp:
                path = BytesIO(temp.read(
                    max(temp.namelist())))  # max chosen so the exact same acc and gyr files are selected each time (repeatability)
        return path

    def rwhar_load_csv(path):
        # Loads up the csv at given path, returns a dictionary of data at each location

        path = check_rwhar_zip(path)
        tables_dict = {}
        with zipfile.ZipFile(path, "r") as Zip:
            zip_files = Zip.namelist()

            for csv in zip_files:
                if "csv" in csv:
                    location = RWHAR_BAND_LOCATION[
                        csv[csv.rfind("_") + 1:csv.rfind(".")]]  # location is between last _ and .csv extension
                    sensor = csv[:3]
                    prefix = sensor.lower() + "_"
                    table = pd.read_csv(Zip.open(csv))
                    table.rename(columns={"attr_x": prefix + "x",
                                          "attr_y": prefix + "y",
                                          "attr_z": prefix + "z",
                                          "attr_time": "timestamp",
                                          }, inplace=True)
                    table.drop(columns="id", inplace=True)
                    tables_dict[location] = table

        return tables_dict

    def rwhar_load_table_activity(path_acc):
        # Logic for loading each activity zip file for acc and gyr and then merging the tables at each location

        acc_tables = rwhar_load_csv(path_acc)
        data = pd.DataFrame()

        for loc in acc_tables.keys():
            acc_tab = acc_tables[loc]

            acc_tab = pd.DataFrame(acc_tab)
            acc_tab["location"] = loc

            data = data.append(acc_tab)

        return data

    def clean_rwhar(filepath, sel_location=None):
        # the function reads the files in RWHAR dataset and each subject and each activity labelled in a panda table
        # filepath is the parent folder containing all the RWHAR dataset.
        # Note: all entries are loaded but their timestamps are not syncronised. So a single location must be selected and
        # all entries with NA must be dropped.

        try:
            subject_dir = os.listdir(filepath)
        except NotADirectoryError:
            try:
                subject_dir = os.listdir(filepath[:-4])
                filepath = filepath[:-4]
            except (NotADirectoryError, FileNotFoundError):
                zf = zipfile.ZipFile(filepath)
                filepath = filepath[:-4]  # Remove the .zip extension
                zf.extractall(path=filepath)
                subject_dir = os.listdir(filepath)

        dataset = pd.DataFrame()

        for sub in subject_dir:
            if "proband" not in sub:
                continue
            #         files = os.listdir(filepath+sub)
            #         files = [file for file in files if (("acc" in file or "gyr" in file) and "csv" in file)]
            subject_num = int(sub[7:]) - 1  # proband is 7 letters long so subject num is number following that
            sub_pd = pd.DataFrame()

            for activity in RWHAR_ACTIVITY_NUM.keys():  # pair the acc and gyr zips of the same activity
                activity_name = "_" + activity + "_csv.zip"
                path_acc = filepath + '/' + sub + "/data/acc" + activity_name  # concat the path to acc file for given activity and subject
                table = rwhar_load_table_activity(path_acc)

                table["activity"] = RWHAR_ACTIVITY_NUM[activity]  # add a activity column and fill it with activity num
                sub_pd = sub_pd.append(table)

            sub_pd["subject"] = subject_num  # add subject id to all entries
            dataset = dataset.append(sub_pd)
            dataset = dataset[dataset.location == RWHAR_BAND_LOCATION[sel_location]]

        if sel_location is not None:
            print("Selecting location : ", sel_location)
            dataset = dataset[dataset.location == RWHAR_BAND_LOCATION[sel_location]]
            dataset = dataset.drop(columns="location")

        dataset = dataset.sort_values(by=['subject', 'timestamp'])
        dataset = dataset.drop(columns="timestamp")
        print(dataset['activity'].value_counts())
        return dataset

    data = clean_rwhar(f'{raw_dir}/realworld2016_dataset.zip', sel_location='forearm')

    return data