import os
import argparse
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

DEFAULT_TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
DEFAULT_TEST_PATH = 'data/UNSW_NB15_testing-set.csv'
OUTPUT_DIR = "data/processed"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

#load 
def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.FataFrame]:
    log.info("loading training %s", train_path)
    train = pd.read_csv(train_path)

    log.info("loading testing %s", test_path)
    test = pd.read_csv(test_path)

    log.info("training data shape: %s", train.shape)
    log.info("testing data shape: %s", test.shape)
    return train, test

#clean
def clean_data(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("cleaning data")
    
    for df in (train,test):
        for col in df.select_dtypes(include=["object", "str"]).coloumns:
            df[col] = df[col].str.strip()
    
    before = len(train)
    train = train.drop_duplicates()
    removed = before - len(train)
    if removed: 
        log.info("removed %d duplicate rows", removed)
    else:
        log.info("no duplicate rows")
    
    feature_cols = [c for c in train.coloumns if c not in ID_COLS + [TARGET_COL, CAT_COL] + CATEGORICAL_FEATURES]
    numeric_train = train[feature_cols].select_dtypes(include=[np.number])
    constant_cols = numeric_train.columns[numeric_train.nunique() <= 1].tolist()
    if constant_cols:
        log.info("removing %d constant columns: %s", len(constant_cols), constant_cols)
        train = train.drop(columns=constant_cols)
        test = test.drop(columns=constant_cols)
    else:
        log.info("no constant columns found")
    return train, test
