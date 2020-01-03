#!/usr/bin/env python
# -*- coding: utf-8 -*-
# palm_utils.py
"""
Some utilities for PALM reconstruction.

Copyright (c) 2016, David Hoffman
"""

import numpy as np
import pandas as pd
from scipy.io import readsav


def peakselector_df(path, verbose=False):
    """Read a peakselector file into a pandas dataframe"""
    print("Reading {} into memory ... ".format(path))
    sav = readsav(path, verbose=verbose)
    df = pd.DataFrame(
        sav["cgroupparams"].byteswap().newbyteorder(), columns=sav["rownames"].astype(str)
    )
    return df


def grouped_peaks(df):
    """Return a DataFrame with only grouped peaks."""
    return df[df["Frame Index in Grp"] == 1]


class PALMData(object):
    """A simple class to manipulate peakselector data"""

    # columns we want to keep
    # 'amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'rho', 'offset'
    peak_col = {
        "X Position": "xpos",
        "Y Position": "ypos",
        "6 N Photons": "nphotons",
        "Frame Number": "framenum",
        "Sigma X Pos Full": "sigmax",
        "Sigma Y Pos Full": "sigmay",
        "Z Position": "zpos",
        "Offset": "offset",
        "Amplitude": "amp",
    }

    group_col = {
        "Frame Number": "framenum",
        "Group X Position": "xpos",
        "Group Y Position": "ypos",
        "Group Sigma X Pos": "sigmax",
        "Group Sigma Y Pos": "sigmay",
        "Group N Photons": "nphotons",
        "24 Group Size": "groupsize",
        "Group Z Position": "zpos",
        "Offset": "offset",
        "Amplitude": "amp",
    }

    def __init__(self, path_to_sav, *args, verbose=True, init=False, **kwargs):
        """To initialize the experiment we need to know where the raw data is
        and where the peakselector processed data is
        
        Assumes paths_to_raw are properly sorted"""

        # load peakselector data
        raw_df = peakselector_df(path_to_sav, verbose=verbose)
        # convert Frame number to int
        raw_df["Frame Number"] = raw_df["Frame Number"].astype(int)
        self.processed = raw_df[list(self.peak_col.keys())]
        self.grouped = grouped_peaks(raw_df)[list(self.group_col.keys())]
        # normalize column names
        self.processed = self.processed.rename(columns=self.peak_col)
        self.grouped = self.grouped.rename(columns=self.group_col)
        # initialize filtered ones
        self.processed_filtered = None
        self.grouped_filtered = None

    def filter_peaks(self, offset=1000, sigma_max=3, nphotons=0, groupsize=5000):
        """Filter internal dataframes"""
        for df_title in ("processed", "grouped"):
            df = self.__dict__[df_title]
            filter_series = (
                (df.offset > 0)
                & (df.offset < offset)  # we know that offset should be around this value.
                & (df.sigmax < sigma_max)
                & (df.sigmay < sigma_max)
                & (df.nphotons > nphotons)
            )
            if "groupsize" in df.keys():
                filter_series &= df.groupsize < groupsize
            self.__dict__[df_title + "_filtered"] = df[filter_series]

    def hist(self, data_type="grouped", filtered=False):
        if data_type == "grouped":
            if filtered:
                df = self.grouped_filtered
            else:
                df = self.grouped
        elif data_type == "processed":
            if filtered:
                df = self.processed_filtered
            else:
                df = self.processed
        else:
            raise TypeError("Data type {} is of unknown type".format(data_type))
        return df[["offset", "amp", "xpos", "ypos", "nphotons", "sigmax", "sigmay", "zpos"]].hist(
            bins=128, figsize=(12, 12), log=True
        )
