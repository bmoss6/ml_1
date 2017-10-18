import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"



def load_housing_data(housing_path= HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indicies = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indicies]


housing = load_housing_data()
train_set, test_set = split_train_test(housing, .2)
print(len(train_set), "train +", len(test_set), "test")
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)


for train_index, test_index in split.split(housing,housing["income_cat"]):
    start_train_set = housing.loc(train_index)
    start_test_set = housing.loc(test_index)


housing["income_cat"].value_counts()/len(housing)

for set_ in (strat_train_set, strat_test_set):
    set_drop("income_cat", axis =1, inplace=True)

housing = strat_train_set.copy()




