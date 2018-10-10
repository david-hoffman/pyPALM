#!/usr/bin/env python
# -*- coding: utf-8 -*-
# utils.py
"""
Utility functions for pyPALM

Copyright (c) 2018, David Hoffman
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.metrics.classification import *
import logging
logger = logging.getLogger(__name__)

coords = ["z0", "y0", "x0"]


def crop(df, window, shift=False):
    """Crop a palm dataframe

    Parameters
    ----------
    df : pd.dataframe
        DataFrame holding the PALM data, needs to have x0, y0 and z0 columns at least
    window : dict of tuples
        The crop window as a dict of coords, i.e. x0, y0, and/or z0
        with a tuple of cmin, cmax for each dictionary
    shift : bool
        Whether or not to shift the data to the new origin.

    Returns
    -------
    df_cropped : pd.DataFrame
        The PALM data with cropped coordinates

    Example
    -------
    >>> window = dict(x0=(200, 800), y0=(700, 900))
    >>> df_cropped = crop(df, window, True)
    """
    # set up our filter DataFrame
    df_filter = None
    # iterate through the window dict
    all_coords = []
    cmins = []
    for coord, (cmin, cmax) in window.items():
        df_coord = df[coord]
        if df_filter is None:
            df_filter = df_coord > cmin
        else:
            df_filter &= df_coord > cmin
        df_filter &= df_coord < cmax
        if coord in coords:
            # only use coordinates for shifting
            all_coords.append(coord)
            cmins.append(cmin)
            
    new_df = df[df_filter]
    if shift:
        new_df.loc[:, all_coords] -= cmins
    return new_df


def weighted_avg(df, cols=coords, weight="amp"):
    # Do a weighted average of cols, by weight
    # might need to switch the df[weight] to df[weight].values to do multiple weightings
    df = df.dropna()
    temp_mean = df[cols].mul(df[weight], "index")
    temp_var = (df[cols] ** 2).mul(df[weight], "index")
    # calc weighted average
    temp_mean = temp_mean.sum().div(df[weight].sum(), "index")
    temp_std = np.sqrt(temp_var.sum().div(df[weight].sum(), "index") - temp_mean ** 2)
    temp_std.index = ["sigma_" + c for c in temp_std.index]
    result = pd.concat((temp_mean, temp_std))
    result["num_counts"] = len(df)
    return result


def find_outliers(df_in, good, bad, sample_size=300000, classifier=RandomForestClassifier, feature_cols=None, diagnostics=False, **kwargs):
    """Find outlier points by providing example good and bad data

    Parameters
    ----------
    df : pd.dataframe
        DataFrame holding the PALM data, needs to have x0, y0 and z0 columns at least
    good_window : dict of tuples

    bad_window : dict of tuples

    sample_size : int (optional)
        Size of the samples to use for classification
    classifier : sklearn Classifier (optional)
        default is to use the RandomForestClassifier, but any sklearn classifier can be used
    """
    # bad data
    if not isinstance(bad, pd.DataFrame):
        # assume bad is window
        bad = crop(df_in, bad)
    # good data
    if not isinstance(good, pd.DataFrame):
        good = crop(df_in, good)

    # make sure the sample size is reasonable
    if sample_size is not None:
        sample_size = min((len(bad), len(good), sample_size))
        logger.info("Using {} sample size".format(sample_size))
        # sample data and assign "good" column
        bad = bad.sample(n=sample_size).assign(good=0)
        good = good.sample(n=sample_size).assign(good=1)
    else:
        bad = bad.assign(good=0)
        good = good.assign(good=1)

    df = pd.concat([bad, good], ignore_index=True).sample(frac=1.0)  # put them together and then shuffle
    if feature_cols is None:
        # we want to use all columns, so that we can take into account groupsize
        feature_cols = list(df_in.columns)
        # but should be agnostic to position, (may want to leave in z0 if doing it on a slab by slab basis)
        for col in ("x0", "y0", "z0", "frame", "group_id"):
            try:
                feature_cols.remove(col)
            except ValueError:
                pass
    logger.info("Using {}".format(feature_cols))
    # set up training data
    X = df.loc[:, feature_cols]
    y = df.good

    # test train split for diagnostics
    if diagnostics:
        test_size = 0.1
    else:
        test_size = 0.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    def evaluate_preds(y_hat):
        return pd.Series({
            'precision': precision_score(y_test, y_hat),  # true positives
            'recall': recall_score(y_test, y_hat),  # of all good ones
            'f1': f1_score(y_test, y_hat),
            'accuracy': accuracy_score(y_test, y_hat),
            'roc_auc': roc_auc_score(y_test, y_hat)
        })

    def evaluate(clf):
        y_hat = clf.predict(X_test)
        return evaluate_preds(y_hat)

    def feature_importances(clf):
        pd.Series(dict(zip(X.columns, clf.feature_importances_))).sort_index().plot.bar()

    # try to use all cores
    try:
        cl = classifier(n_jobs=-1, **kwargs)
    except TypeError:
        cl = classifier(**kwargs)
    cl.fit(X_train, y_train)

    if diagnostics:
        print(evaluate(cl))
        feature_importances(cl)

    def filter_func(another_df):
        """filter dataframe using a trained classifier"""
        new_df = another_df[cl.predict(another_df[feature_cols]).astype(bool)]
        logger.info("Filtered {}% of peaks".format((1 - len(new_df) / len(another_df)) * 100))
        return new_df

    return filter_func
