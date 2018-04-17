#!/usr/bin/env python
# -*- coding: utf-8 -*-
# grouping.py
"""
All code related to drift correction of PALM data

Copyright (c) 2017, David Hoffman
"""
import psutil
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import dask.dataframe as dd
import dask.multiprocessing
# get a logger
import logging
logger = logging.getLogger(__name__)


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
            # should never get here, see list comprehension below
            raise RuntimeError("Something went wrong in `closest_match`")
        distances = ((t0.data[i] - t1.data[m])**2).sum(1)
        return [i, m[distances.argmin()]]
    
    # return a list with only the closest peak.
    pair_matches = [closest_match(m, i) for i, m in enumerate(pair_matches) if len(m)]
    return pair_matches


def group(df, radius, gap, frame_reset=np.inf):
    """Group peaks based on x y locations

    Parameters
    ----------
    df : pandas DataFrame
    radius : float
    gap : int
    frame_reset : int"""
    new_df_list = []
    # should add a progress bar here
    frame_min = df.frame.min()
    for frame, peaks in df.groupby("frame"):
        peaks = peaks.copy()
        # set/reset group_id
        peaks["group_id"] = -1
        if not (frame - frame_min) % frame_reset:
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


# def agg_groups(df_grouped):
#     def f(x):
#         if len(x) < 2:
#             x = x.drop("group_id", 1)
#             x["groupsize"] = 1
#             return x
        
#         coords = ["x", "y", "z"]
#         x0s = [c + '0' for c in coords]
#         sigmas = ["sigma_" + c for c in coords]
        
#         d = pd.DataFrame(np.nan, columns=x0s + sigmas, index=x.index[:1])
#         inv_sigma = (1 / x[sigmas].values**2).sum(0)
#         d[sigmas] = 1 / np.sqrt(inv_sigma)
        
#         d[x0s] = (x[x0s].values / x[sigmas].values**2).sum(0) / inv_sigma
            
#         d["nphotons"] = x["nphotons"].sum()
#         d["amp"] = x["amp"].sum()
#         d["offset"] = x["offset"].mean()
#         d["groupsize"] = len(x)
#         d["frame"] = x["frame"].iloc[0]

#         return d

#     a = df_grouped.groupby("group_id").apply(f)
#     a.index = a.index.droplevel(1)
#     return a


def agg_groups(df_grouped):
    # define coordinates
    coords = ["x", "y", "z"]
    # save the labels for weighted coords and weights
    w_coords = []
    weights = []
    # loop through coords generating weights and weighted coords
    for c in coords:
        s = "sigma_" + c
        df_grouped[s + "_inv"] = 1 / df_grouped[s] ** 2
        weights.append(s + "_inv")
        x = c + "0"
        df_grouped[x + "_w"] = df_grouped[x].mul(df_grouped[s + "_inv"], "index")
        w_coords.append(x + "_w")
    # groupby group_id and sum
    temp_gb = df_grouped.groupby("group_id")
    # finish weighted mean
    new_coords = temp_gb[w_coords].sum() / temp_gb[weights].sum().values
    new_coords.columns = [c.replace("_w", "") for c in new_coords.columns]
    # calc new sigma
    new_sigmas = np.sqrt(1 / temp_gb[weights].sum())
    new_sigmas.columns = [c.replace("_inv", "") for c in new_sigmas.columns]
    # calc new group params
    new_amp = temp_gb[["amp", "nphotons", "chi2"]].sum()
    new_frame = temp_gb[["frame"]].first()
    groupsize = temp_gb.x0.count()
    groupsize.name = "groupsize"
    new_offset = temp_gb[["offset"]].mean()
    # drop added columns from original data frame
    df_grouped.drop(columns=w_coords + weights, inplace=True)
    # return new data frame
    return pd.concat([new_coords, new_sigmas, new_amp, new_frame, groupsize, new_offset], axis=1)


def measure_peak_widths(y):
    """Measure peak widths in thresholded data. 

    Parameters
    ----------
    y : iterable (ndarray, 1d)
        binary data

    Returns
    -------
    widths : ndarray, 1d
        Measured widths of the peaks.
    """
    d = np.diff(y)
    i = np.arange(len(d))
    rising_edges = i[d > 0]
    falling_edges = i[d < 0]
    # need to deal with all cases
    # same number of edges
    if len(rising_edges) == len(falling_edges):
        if len(rising_edges) == 0:
            return 0
        # starting and ending with peak
        # if falling edge is first we remove it
        if falling_edges.min() < rising_edges.min():
            widths = np.append(falling_edges, i[-1]) - np.append(0, rising_edges)
        else:
            # only peaks in the middle
            widths = falling_edges - rising_edges
    else:
        # different number of edges
        if len(rising_edges) < len(falling_edges):
            # starting with peak
            widths = falling_edges - np.append(0, rising_edges)
        else:
            # ending with peak
            widths = np.append(falling_edges, i[-1]) - rising_edges
    return widths


def count_blinks(offtimes, gap):
    """Count the number of blinkers based on offtimes and a fixed gap

    ontimes = measure_peak_widths((y > 0) * 1
    offtimes = measure_peak_widths((y == 0) * 1

    """
    breaks = np.nonzero(offtimes > gap)[0]
    if breaks.size:
        blinks = [offtimes[breaks[i] + 1:breaks[i+1]] for i in range(breaks.size - 1)]
    else:
        blinks = [offtimes]
    return ([len(blink) for blink in blinks])


"""
temp_data = palm.processed_filtered
titles = ("On Times", "Off Times", "Number of Blinks")
bins = (np.arange(128), np.arange(251), np.arange(60))
for radius in (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)[::-1]:
    # extract samples
    more_samples_blinks = extract_fiducials(temp_data, more_samples[["y0", "x0"]].values, radius)
    temp_data = pd.concat(more_samples_blinks)
    # calculate localizations per sample
    print("Calculating localizations ... ")
    %time more_samples_localizations = np.array([f.groupby("frame").count().x0.reindex(np.arange(max_frame)).fillna(0).astype(int) for f in more_samples_blinks])
    # calculate mean positions
    print("Calculating mean positions ... ")
    %time more_samples_positions = [f.mean() for f in more_samples_blinks]
    # calculate on times
    print("Calculating on times ... ")
    %time more_samples_ontimes = np.concatenate([measure_peak_widths((y > 0) * 1) for y in more_samples_localizations if y.sum() < 0.1 * max_frame])
    # calculate off times
    print("Calculating off times ... ")
    %time offtimes = [measure_peak_widths((y == 0) * 1) for y in more_samples_localizations if y.sum() < 0.1 * max_frame]
    more_samples_offtimes = np.concatenate(offtimes)
    more_samples_offtimes2 = more_samples_offtimes[more_samples_offtimes < 251]
    
    a = np.histogram(more_samples_offtimes2,  bins[1])
    b = a[0].cumsum() / a[0].cumsum().max()
    group_gap = (np.abs(b - 0.9)).argmin()
    
    blinks = np.concatenate([count_blinks(s, group_gap) for s in offtimes])

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    to_plot = (more_samples_ontimes, more_samples_offtimes2, blinks)
    
    for ax, p, t, b in zip(axs.ravel(), to_plot, titles, bins):
        ax.hist(p, log=False, bins=b);
        ax1 = ax.twinx()
        ax1.hist(p, log=True, bins=b, color="r");
        ax1.set_zorder(-1)
        ax1.tick_params(axis='y', colors='red')
        ax.patch.set_facecolor((1,1,1,0))
        ax.set_title(t)
    
    axs[1].axvline(group_gap, color="g")
    
    t = "Halo-TOMM20 (JF549) 4K Blinking, amp=50, radius = {:.2f}, gap = {}".format(radius, group_gap)
    fig.suptitle(t, y=1)
    fig.tight_layout()
    fig.savefig(t + ".png".format(radius), dpi=300, bbox_inches="tight")
"""