import os
import numpy as np
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


def save_proxy_data(dic):
    f = open('proxy_data.txt', 'w')
    f.write(str(dic))
    f.close()


def load_proxy_data():
    f = open('proxy_data.txt', 'r')
    data = f.read()
    f.close()
    return eval(data)


def bmr(sex, weight, height, age):
    """
        return the Basal Metabolic Rate (BMR)
        based on Miffin-St Jeor Equation
    :param sex: person sex, m or w
    :param weight: person weight in kg
    :param height: person height in cm
    :param age: person age in years
    :return: bmr
    """
    base_bmr = 10 * weight + 6.25 * height - 5 * age
    return base_bmr + 5 if sex == 'm' else base_bmr - 161


def total_intake_calories(bmr, activity_level):
    """
        Calculate the Total Intake Calories based
        on activity level
    :param bmr: value from bmr function
    :param activity_level: reference level, 0 - sedentary, 1 - low, 2 - moderate, 3 - high
    :return: average target intake calories
    """
    ref_activities = [1.2, 1.4, 1.65, 1.95]
    return bmr * ref_activities[activity_level]


def general_intake_by_group(tic):
    return tic * np.array([0.4, 0.3, 0.3])
