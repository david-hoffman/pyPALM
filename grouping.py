#!/usr/bin/env python
# -*- coding: utf-8 -*-
# grouping.py
"""
All code related to drift correction of PALM data

Copyright (c) 2017, David Hoffman
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def find_matches(frames, radius):
    """Find matching peaks between two frames

    Parameters
    ----------
    frames : iterable of ndarrays (2, n) of len 2
        The peak locations in space
    radius : float
        the radius within which to consider peaks matched

    Returns
    -------
    pair_matches : list of lists
        A list of closest matches
    """
    frame0, frame1 = frames
    t0 = cKDTree(frame0)
    t1 = cKDTree(frame1)
    pair_matches = t0.query_ball_tree(t1, radius)
    # return only the closest match.
    
    def closest_match(m, i):
        if len(m) == 1:
            return [i, m[0]]
        elif len(m) == 0:
            # should never get here
            raise RuntimeError("Something went wrong in `closest_match`")
        distances = ((t0.data[i] - t1.data[m])**2).sum(1)
        return [i, m[distances.argmin()]]
    
    # return a list with only the closest peak.
    pair_matches = [closest_match(m, i) for i, m in enumerate(pair_matches) if len(m)]
    return pair_matches


def group(df, radius, gap):
    """Group peaks based on x y locations"""
    new_df_list = []
    for frame, peaks in df.groupby("frame"):
        peaks = peaks.copy()
        # set/reset group_id
        peaks["group_id"] = -1
        if frame == 0:
            # group_id will be the index of the first peak
            df_cache = peaks
            df_cache.loc[peaks.index, "group_id"] = df_cache.index
            new_df_list.append(df_cache.copy())
            continue
        # search for matches
        matches = find_matches([df_cache[["y0", "x0"]].values, peaks[["y0", "x0"]].values], radius)
        # get indices
        # need to deal with overlaps (two groups claim same peak)
        try:
            # if there is a new peak that matches to two or more different cached peaks then the newer of the
            # cached peaks claims it. If the cached peaks have the same age then its a toss up.
            cache_idx, peaks_idx = np.array([[df_cache.index[i], peaks.index[m]] for i, m in matches]).T
        except ValueError as error:
            # should log the error or raise as a warning.
            pass
        else:
            # update groups
            # need to use .values, because list results in DF
            peaks.loc[peaks_idx, "group_id"] = df_cache.loc[cache_idx, "group_id"].values
        # ungrouped peaks get their own group_id
        peaks.group_id.where((peaks.group_id != -1), peaks.index, inplace=True)
        # peaks.loc[(peaks.group_id != -1), "group_id"] = peaks.index
        # update df_cache and lifetimes
        # updating the cache takes a significant amount of time.
        df_cache = pd.concat((df_cache, peaks))
        df_cache = df_cache.drop_duplicates("group_id", "last")
        df_cache = df_cache[(frame - df_cache.frame) < gap]
        new_df_list.append(peaks)
    return pd.concat(new_df_list)
