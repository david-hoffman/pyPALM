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
    # backwards looking, find points in frame0 that match points in frame1
    pair_matches = t1.query_ball_tree(t0, radius)
    # return only the closest match.
    
    def closest_match(m, i):
        """i is index in frame1, m is index in frame0
        make sure to return frame0 index first
        """
        if len(m) == 1:
            return [m[0], i]
        elif len(m) == 0:
            # should never get here, see list comprehension below
            raise RuntimeError("Something went wrong in `closest_match`")
        distances = ((t1.data[i] - t0.data[m])**2).sum(1)
        # return [frame0_idx, frame1_idx]
        return [m[distances.argmin()], i]
    
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
    frame_reset : int
    """
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
        if len(matches):
            try:
                # if there is a new peak that matches to two or more different cached peaks then the newer of the
                # cached peaks claims it. If the cached peaks have the same age then its a toss up.
                cache_idx, peaks_idx = np.array([[df_cache.index[i], peaks.index[m]] for i, m in matches]).T
            except ValueError as error:
                # should log the error or raise as a warning.
                logger.warning(error)
            else:
                # update groups
                # need to use .values, because list results in DF
                peaks.loc[peaks_idx, "group_id"] = df_cache.loc[cache_idx, "group_id"].values
        # ungrouped peaks get their own group_id
        peaks.group_id.where((peaks.group_id != -1), peaks.index.values, inplace=True)
        # peaks.loc[(peaks.group_id != -1), "group_id"] = peaks.index
        # update df_cache and lifetimes
        # updating the cache takes a significant amount of time.
        df_cache = pd.concat((df_cache, peaks))
        # the commented line below is more accurate but about 4-5X slower
        # df_cache = agg_groups(df_cache).reset_index()
        df_cache = df_cache.drop_duplicates("group_id", "last")
        df_cache = df_cache[(frame - df_cache.frame) < gap]
        new_df_list.append(peaks)
    return pd.concat(new_df_list)


def agg_groups(df_grouped):
    """Aggregate groups
    
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    """
    coords = ["x", "y", "z"]
    # save the labels for weighted coords and weights
    w_coords = []
    w_coords2 = []
    weights = []
    # loop through coords generating weights and weighted coords
    for c in coords:
        # calculate weights
        s = "sigma_" + c
        df_grouped[s + "_inv"] = 1 / df_grouped[s] ** 2
        weights.append(s + "_inv")
        x = c + "0"
        # weighted position
        df_grouped[x + "_w"] = df_grouped[x].mul(df_grouped[s + "_inv"], "index")
        w_coords.append(x + "_w")
        # weighted position^2
        df_grouped[x + "_w2"] = (df_grouped[x]**2).mul(df_grouped[s + "_inv"], "index")
        w_coords2.append(x + "_w2")
    # groupby group_id and sum
    temp_gb = df_grouped.groupby("group_id")
    # calculate sum weights
    sum_w = temp_gb[weights].sum().values
    # finish weighted mean
    new_coords = temp_gb[w_coords].sum() / sum_w
    # doing this here to preserve order
    new_coords2_values = (new_coords**2)[w_coords].values
    new_coords.columns = [c.replace("_w", "") for c in new_coords.columns]
    # calc new sigma
    # basically the weighted std dev of the points added in quadrature to the 
    # std dev of the std dev of all points
    # $$
    # \sqrt{\frac{\sum w_i (x_i - \mu^*)^2}{\sum w_i} + \frac{1}{\sum w_i}} \\
    # \sqrt{\frac{\sum w_i x_i^2}{\sum w_i} - \mu^{*2} + \frac{1}{\sum w_i}} \\
    # \sqrt{\frac{\sum w_i x_i^2 + 1}{\sum w_i} - \mu^{*2}} \\
    # w_i = \frac{1}{\sigma_i^2}
    # $$
    new_sigmas = np.sqrt((temp_gb[w_coords2].sum() + 1) / sum_w -
                         new_coords2_values)
    new_sigmas.columns = ["sigma_" + c[0] for c in new_sigmas.columns]
    # calc new group params
    new_amp = temp_gb[["amp", "nphotons", "chi2"]].sum()
    new_frame = temp_gb[["frame"]].last()
    groupsize = temp_gb.x0.count()
    groupsize.name = "groupsize"
    # take the mean of all remaining columns
    other_columns = ["groupsize"] + w_coords + w_coords2 + weights
    for df in (new_coords, new_sigmas, new_amp, new_frame):
        other_columns += df.columns.tolist()
    columns_to_mean = df_grouped.columns.difference(other_columns)
    new_means = temp_gb[columns_to_mean].mean()
    # drop added columns from original data frame
    df_grouped.drop(columns=w_coords + weights + w_coords2, inplace=True)
    # return new data frame
    return pd.concat([new_coords, new_sigmas, new_amp, new_frame, groupsize, new_means], axis=1)


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
        blinks = [offtimes[breaks[i] + 1:breaks[i + 1]] for i in range(breaks.size - 1)]
    else:
        blinks = [offtimes]
    return ([len(blink) for blink in blinks])


"""
temp_data = mito_data
titles = ("On Times", "Off Times", "Number of Blinks")
bins = (np.arange(128), np.arange(512), np.arange(60))
for radius in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
    # extract samples
    %time more_samples_blinks = pdiag.extract_fiducials(temp_data, more_samples[["y0", "x0"]].values, radius, True)
    print(len(temp_data))
#     temp_data = pd.concat(more_samples_blinks)
    print(len(temp_data))
    # calculate localizations per sample

    print("Calculating localizations ... ")
    %time more_samples_localizations = np.array([f.groupby("frame").count().x0.reindex(np.arange(max_frame)).fillna(0).astype(int) for f in more_samples_blinks])
    # clear ones with too much density to be reliable, ones with more than 10,000 on frames
    # and ones without any
    total_localizations = more_samples_localizations.sum(1)
    more_samples_localizations = more_samples_localizations[(total_localizations < 1e4) & (total_localizations > 0)]
    print("Kept samples = {}%".format(int(len(more_samples_localizations) / len(total_localizations) * 100)))
    # calculate mean positions
    # print("Calculating mean positions ... ")
    # %time more_samples_positions = [f.mean() for f in more_samples_blinks]
    # calculate on times
    print("Calculating on times ... ")
    # make sure the thing isn't on for more than 10000 frames, otherwise area is too dense
    %time ontimes = [pdiag.measure_peak_widths((y > 0) * 1) for y in more_samples_localizations]
    more_samples_ontimes = np.concatenate(ontimes)
    # calculate off times
    print("Calculating off times ... ")
    %time offtimes = [pdiag.measure_peak_widths((y == 0) * 1) for y in more_samples_localizations]
    more_samples_offtimes = np.concatenate(offtimes)
    # when is the first time the histogram hits 0? That's our noise floor
    noise_floor = np.bincount(more_samples_offtimes)[1:].argmin() + 1
    print("noise_floor", noise_floor)
    more_samples_offtimes2 = more_samples_offtimes#[more_samples_offtimes <= noise_floor]
    
    group_gap = int(np.percentile(more_samples_offtimes2, 90))
    print("Group gap", group_gap)
    p_100 = (more_samples_offtimes2 <= 100).sum() / more_samples_offtimes2.size
    
    blinks = np.concatenate([pdiag.count_blinks(s, group_gap) for s in offtimes])
    
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    to_plot = (more_samples_ontimes, more_samples_offtimes2, blinks)
        
    for ax, p, t, b in zip(axs.ravel(), to_plot, titles, bins):
        p = np.clip(p, b[0], b[-1])
        ax.hist(p, log=False, bins=b, histtype='stepfilled', color="lightblue", density=True);
        ax.hist(p, log=False, bins=b, histtype='step', cumulative=True, color="blue", density=True);
        ax1 = ax.twinx()
        ax1.hist(p, log=True, bins=b, histtype='stepfilled', color="coral", density=True);
        ax1.hist(p, log=True, bins=b, histtype='step', cumulative=True, color="red", density=True);
        ax1.set_zorder(-1)
        ax1.tick_params(axis='y', colors='red')
        ax.patch.set_facecolor((1,1,1,0))
        ax.set_title(t)
    
    if b[-1] > group_gap:
        axs[1].axvline(group_gap, color="g")
    axs[1].scatter([100], [p_100], label="{}% / 100".format(int(p_100 * 100)))
    axs[1].legend(loc="lower right")
     
    t = "Halo-TOMM20 (JF525) 4K Blinking, Singles, radius = {:.2f}, gap = {}".format(radius, group_gap)
    fig.suptitle(t, y=1)
    fig.tight_layout()
    #break
    fig.savefig(t + ".png".format(radius), dpi=300, bbox_inches="tight")
"""
