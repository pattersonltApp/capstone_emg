import os
import pandas as pd
import numpy as np


def reformat_files(directory):
    label = '*'
    sample_num = 0
    for filename in os.listdir(directory):
        if filename[0] == label:
            sample_num += 1
        else:
            sample_num = 0
        label = filename[0]

        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            df = pd.read_csv(f)
            df = df.drop(['time (ms)', 'time (sec)'], axis=1)
            df = df.transpose()
            new_fname = directory + '/cleaned_data/' + label + '_' + str(sample_num) + '.csv'
            df.to_csv(new_fname, index=False, header=False)


if __name__ == '__main__':
    dirs = ['BA_data/raw_data', 'MJ_data/raw_data']
    for dir in dirs:
        reformat_files(dir)
