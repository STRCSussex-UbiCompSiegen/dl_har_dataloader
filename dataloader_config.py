##################################################
# Config objects for all the different datasets. Contain all relevant metadata.
##################################################
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
# Author: Marius Bock
# Email: marius.bock@uni-siegen.de
##################################################

import numpy as np

user_1_files = ['S1-Drill', 'S1-ADL1', 'S1-ADL2', 'S1-ADL3', 'S1-ADL4', 'S1-ADL5']
user_2_files = ['S2-Drill', 'S2-ADL1', 'S2-ADL2', 'S2-ADL3', 'S2-ADL4', 'S2-ADL5']
user_3_files = ['S3-Drill', 'S3-ADL1', 'S3-ADL2', 'S3-ADL3', 'S3-ADL4', 'S3-ADL5']
user_4_files = ['S4-Drill', 'S4-ADL1', 'S4-ADL2', 'S4-ADL3', 'S4-ADL4', 'S4-ADL5']
opportunity_config = {'name': 'opportunity',
                      'data_archive': 'OpportunityUCIDataset.zip',
                      'data_files': {0: ['OpportunityUCIDataset/dataset/{}.dat'.format(run) for run in user_1_files],
                                     1: ['OpportunityUCIDataset/dataset/{}.dat'.format(run) for run in user_2_files],
                                     2: ['OpportunityUCIDataset/dataset/{}.dat'.format(run) for run in user_3_files],
                                     3: ['OpportunityUCIDataset/dataset/{}.dat'.format(run) for run in user_4_files]},
                      'class_map': ["Null",
                                    "Open Door 1",
                                    "Open Door 2",
                                    "Close Door 1",
                                    "Close Door 2",
                                    "Open Fridge",
                                    "Close Fridge",
                                    "Open Dishwasher",
                                    "Close Dishwasher",
                                    "Open Drawer 1",
                                    "Close Drawer 1",
                                    "Open Drawer 2",
                                    "Close Drawer 2",
                                    "Open Drawer 3",
                                    "Close Drawer 3",
                                    "Clean Table",
                                    "Drink from Cup",
                                    "Toggle Switch", ],
                      'n_classes': 18,
                      'n_channels': 113,
                      'label_column': 249,
                      'sensor_columns': np.concatenate(
                          [np.arange(1, 46),
                           np.arange(50, 59),
                           np.arange(63, 72),
                           np.arange(76, 85),
                           np.arange(89, 98),
                           np.arange(102, 134)]),
                      'sr': 32,
                      'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip',
                      'down_sample': False,
                      'exclude_channels': np.concatenate(
                          [np.arange(46, 50),
                           np.arange(59, 63),
                           np.arange(72, 76),
                           np.arange(85, 89),
                           np.arange(98, 102),
                           np.arange(134, 249)]),
                      'label_map': {
                          406516: 1, 406517: 2, 404516: 3, 404517: 4, 406520: 5, 404520: 6,
                          406505: 7, 404505: 8, 406519: 9,
                          404519: 10, 406511: 11, 404511: 12, 406508: 13, 404508: 14, 408512: 15,
                          407521: 16, 405506: 17},
                      'delimiter': ' '}

path1 = f'PAMAP2_Dataset/Protocol/subject'
path2 = f'PAMAP2_Dataset/Optional/subject'

subjects = ['101', '102', '103', '104', '105', '106', '107', '108', '109']

data_files = {i: [f'{path1}{sub}.dat', f'{path2}{sub}.dat'] for i, sub in enumerate(subjects)}

pamap2_config = {'name': 'pamap2',
                 'data_archive': 'PAMAP2_Dataset.zip',
                 'data_files': data_files,
                 'sr': 98,
                 'class_map': ["Rope Jumping",
                               "Lying",
                               "Sitting",
                               "Standing",
                               "Walking",
                               "Running",
                               "Cycling",
                               "Nordic Walking",
                               "Ascending Stairs",
                               "Descending Stairs",
                               "Vacuum Cleaning",
                               "Ironing",
                               ],
                 'n_classes': 19,
                 'n_channels': 52,
                 'label_column': 1,
                 'sensor_columns': np.arange(2, 54),
                 'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip',
                 'down_sample': 3,
                 'label_map': {24: 0,
                               12: 8,
                               13: 9,
                               16: 10,
                               17: 11,
                               },
                 'exclude_labels': [0, 9, 10, 11, 18, 19, 20],
                 'delimiter': ' '}

# SHL config - WIP
# zip1_path1 = f'SHLDataset_preview_v1/User1/220617'
# zip1_path2 = f'SHLDataset_preview_v1/User1/260617'
# zip1_path3 = f'SHLDataset_preview_v1/User1/270617'
#
# zip2_path1 = f'SHLDataset_preview_v1/User2/140617'
# zip2_path2 = f'SHLDataset_preview_v1/User2/140717'
# zip2_path3 = f'SHLDataset_preview_v1/User2/180717'
#
# zip3_path1 = f'SHLDataset_preview_v1/User3/030717'
# zip3_path2 = f'SHLDataset_preview_v1/User3/070717'
# zip3_path3 = f'SHLDataset_preview_v1/User3/140617'
#
# positions = ['Bag', 'Hand', 'Hips', 'Torso']
#
# data_files_shl = {0: [(0, f'{zip1_path1}/Hips_Motion.txt'),
#                       (0, f'{zip1_path2}/Hips_Motion.txt'),
#                       (0, f'{zip1_path3}/Hips_Motion.txt')],
#                   1: [(1, f'{zip2_path1}/Hips_Motion.txt'),
#                       (1, f'{zip2_path2}/Hips_Motion.txt'),
#                       (1, f'{zip2_path3}/Hips_Motion.txt')],
#                   2: [(2, f'{zip3_path1}/Hips_Motion.txt'),
#                       (2, f'{zip3_path2}/Hips_Motion.txt'),
#                       (2, f'{zip3_path3}/Hips_Motion.txt')]}
#
# label_files_shl = {0: [(0, f'{zip1_path1}/Label_Motion.txt'),
#                        (0, f'{zip1_path2}/Label_Motion.txt'),
#                        (0, f'{zip1_path3}/Label_Motion.txt')],
#                    1: [(1, f'{zip2_path1}/Label_Motion.txt'),
#                        (1, f'{zip2_path2}/Label_Motion.txt'),
#                        (1, f'{zip2_path3}/Label_Motion.txt')],
#                    2: [(2, f'{zip3_path1}/Label_Motion.txt'),
#                        (2, f'{zip3_path2}/Label_Motion.txt'),
#                        (2, f'{zip3_path3}/Label_Motion.txt')]}
#
# shl_preview_config = {'name': 'shlpreview',
#                       'data_dir': '../data/raw',
#                       'data_archive': ['SHLDataset_preview_v1_part1.zip',
#                                        'SHLDataset_preview_v1_part2.zip',
#                                        'SHLDataset_preview_v1_part3.zip'],
#                       'data_files': data_files_shl,
#                       'sr': 100,
#                       'class_map': ['Null',
#                                     'Still',
#                                     'Walking',
#                                     'Run',
#                                     'Bike',
#                                     'Car',
#                                     'Bus',
#                                     'Train',
#                                     'Subway'],
#                       'n_classes': 9,
#                       'n_channels': 88,
#                       'label_column': 1,
#                       'sensor_columns': np.arange(2, 24),
#                       'url': ['http://www.shl-dataset.org/wp-content/uploads/SHLDataset_preview_v1_part1.zip',
#                               'http://www.shl-dataset.org/wp-content/uploads/SHLDataset_preview_v1_part2.zip',
#                               'http://www.shl-dataset.org/wp-content/uploads/SHLDataset_preview_v1_part3.zip'],
#                       'down_sample': 3,
#                       'label_files': label_files_shl,
#                       'n_channels_per_file': 22}

hhar_config = {'name': 'hhar',
               'data_archive': 'Activity Recognition exp.zip',
               'data_files': {i: [r'Activity recognition exp/Watch_accelerometer.csv'] for i in range(9)},
               'class_map': {0: 'Null',
                             1: 'Sit',
                             2: 'Stand',
                             3: 'Walk',
                             4: 'Ascend Stairs',
                             5: 'Descend Stairs',
                             6: 'Bike'},
               'n_channels': 12,
               'n_classes': 7,
               'label_column': 'gt',
               'sensor_columns': ['x', 'y', 'z'],
               'user_column': 'User',
               'user_map': {'a': 0,
                            'b': 1,
                            'c': 2,
                            'd': 3,
                            'e': 4,
                            'f': 5,
                            'g': 6,
                            'h': 7,
                            'i': 8
                            },
               'sr': 100,
               'down_sample': 3,
               'label_map': {'null': 0,
                             'sit': 1,
                             'stand': 2,
                             'walk': 3,
                             'stairsup': 4,
                             'stairsdown': 5,
                             'bike': 6},
               'url': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip',
               'n_channels_per_file': 3,
               'header': 0,
               'delimiter': ','}

rwhar_config = {'name': 'rwhar',
                'data_archive': 'realworld2016_dataset.zip',
                'data_files': {1:  None,
                               2:  None,
                               3:  None,
                               4:  None,
                               5:  None,
                               6:  None,
                               7:  None,
                               8:  None,
                               9:  None,
                               10: None,
                               11: None,
                               12: None,
                               13: None,
                               14: None,
                               15: None},
                'class_map': {'climbing_down': 0,
                              'climbing_up': 1,
                              'jumping': 2,
                              'lying': 3,
                              'running': 4,
                              'sitting': 5,
                              'standing': 6,
                              'walking': 7},
                'n_classes': 8,
                'n_channels': 3,
                'label_column': 'activity',
                'sensor_columns': ['acc_x', 'acc_y', 'acc_z'],
                'sr': 100,
                'down_sample': 3,
                'url': 'http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip'}

CONFIGS = {'opportunity': opportunity_config,
           'pamap2': pamap2_config,
           # 'shlpreview': shl_preview_config,
           'hhar': hhar_config,
           'rwhar': rwhar_config}
