import pandas as pd


def read_srleg(path, names):
    return pd.read_table(path, sep='^', na_values='^^|~~', header=None, names=names,
                         quotechar='~', encoding='Latin-1')
