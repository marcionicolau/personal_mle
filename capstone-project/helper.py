import os

import pandas as pd


def read_srleg(path, names):
    return pd.read_table(path, sep='^', na_values='^^|~~', header=None, names=names,
                         quotechar='~', encoding='Latin-1')


def convert_to_csv(input_path, files, names, output_path='Datasets'):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ds = {}

    for f in files:
        from_f = os.path.join(input_path, f)
        filename, file_extension = os.path.splitext(f)
        to_f = os.path.join(output_path, filename + ".csv")
        temp = read_srleg(from_f, names[f])
        ds[filename] = temp
        temp.to_csv(to_f, index=False)

    return ds
