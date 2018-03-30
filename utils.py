#!/usr/bin/env python
# -*- coding: utf-8 -*-
# utils.py
"""
Utility functions for pyPALM

Copyright (c) 2018, David Hoffman
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
        all_coords.append(coord)
        cmins.append(cmin)

    assert set(all_coords) == set(window.keys()), "all_coords doesn't equal input coords"
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


def find_outliers(df_in, good_window, bad_window, sample_size=300000, classifier=RandomForestClassifier):
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
    bad = crop(df_in, bad_window)
    # good data
    good = crop(df_in, good_window)

    # make sure the sample size is reasonable
    sample_size = min((len(bad), len(good), sample_size))
    
    # sample data and assign "good" column
    bad = bad.sample(n=sample_size).assign(good=0)
    good = good.sample(n=sample_size).assign(good=1)

    df = pd.concat([bad, good]).sample(frac=1.0)  # put them together and then shuffle

    feature_cols = ['nphotons', 'sigma_x', 'sigma_y', 'sigma_z', 'offset', 'amp', 'chi2']
    X = df.loc[:, feature_cols]
    y = df.good

    cl = classifier(n_jobs=-1)
    cl.fit(X, y)

    df_classified = df_in.assign(good=cl.predict(df_in[feature_cols]))

    return df_classified
