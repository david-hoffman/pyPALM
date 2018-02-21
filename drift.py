#!/usr/bin/env python
# -*- coding: utf-8 -*-
# drift.py
"""
All code related to drift correction of PALM data

Copyright (c) 2017, David Hoffman
"""

import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from peaks.peakfinder import PeakFinder
from skimage.filters import threshold_otsu
from .render import palm_hist

coords = ["z0", "y0", "x0"]

def remove_xy_mean(df):
    df_new = df.astype(np.float)
    xyz_mean = df_new[coords].mean()
    df_new[coords] -= xyz_mean
    return df_new


def calc_drift(fiducials_df, weighted="amp", diagnostics=False, frames_index=None):
    """Given a list of DataFrames with each DF containing the coordinates
    of a single fiducial calculate the mean or weighted mean of the coordinates
    in each frame."""
    if len(fiducials_df) == 1:
        # if there is only one fiducial then return that
        logger.debug("Only on fiducial passed to calc_drift")
        toreturn = remove_xy_mean(fiducials_df[0])[coords]
    else:
        mean_removed = [remove_xy_mean(ff) for ff in fiducials_df]
        if diagnostics:
            # debugging diagnostics
            fig, (ax0, ax1) = plt.subplots(1, 2)
            for ff in mean_removed:
                ff.x0.plot(ax=ax0)
                ff.y0.plot(ax=ax1)
                
        # want to do a weighted average
        # need to reset_index after concatination so that all localzations have unique ID
        # this will make weighting easier down the line.
        df_means = pd.concat(mean_removed).reset_index()

        # if weighted is something, use that as the weights for the mean
        # if weighted is not a valid column name then it will raise an
        # exception
        if weighted:
            # weight the coordinates
            logger.debug("Weighting by {}".format(weighted))
            df_means[coords] = df_means[coords].mul(df_means[weighted], "index")
            # groupby frame
            temp = df_means.groupby("frame")
            # calc weighted average
            toreturn = temp[coords].sum().div(temp[weighted].sum(), "index")
        else:
            toreturn = df_means.groupby("frame")[coords].mean()

    if frames_index is None:
        return toreturn
    else:
        return toreturn.reindex(frames_index).interpolate(limit_direction="both")




def remove_drift(df_data, drift):
    """Remove the given drift from the data

    Assumes that drift is a dataframe of coordinates indexed by frame."""
    # make our index frame number so that when we subtract drift it aligns automatically along
    # the index, this needs to be tested.
    # this also, conveniently, makes a copy of the data
    df_data_dc = df_data.set_index("frame")
    # subtract drift only (assumes that drift only has these keys)
    df_data_dc[coords] -= drift
    # return the data frame with the index reset so that all localizations have
    # a unique id
    return df_data_dc.reset_index()


def calc_fiducial_stats(fid_df_list):
    """Calculate various stats"""
    fwhm = lambda x: x.std() * (2 * np.sqrt(2 * np.log(2)))
    fid_stats = pd.DataFrame([f[coords + ["amp"]].mean() for f in fid_df_list])
    fid_stats[[c[0] + "drift" for c in coords]] = pd.DataFrame([f.agg({c:fwhm for c in coords}) for
                                                    f in fid_df_list])[coords]
    fid_stats["sigma"] = np.sqrt(fid_stats.ydrift**2 + fid_stats.xdrift**2)
    all_drift = pd.concat([f[["x0","y0", "z0"]] - f[["x0","y0", "z0"]].mean() for f in fid_df_list])
    return fid_stats, all_drift


def extract_fiducials(df, blobs, radius, min_num_frames=0):
    """Do the actual filtering
    
    We're doing it sequentially because we may run out of memory.
    If initial DataFrame is 18 GB (1 GB per column) and we have 200 """
    fiducials_dfs = [df[np.sqrt((df.x0 - x) ** 2 + (df.y0 - y) ** 2) < radius]
        for y, x in tqdm.tqdm_notebook(blobs, leave=False, desc="Extracting Fiducials")]
    # remove any duplicates in a given frame by only keeping the localization with the largest count
    clean_fiducials = [sub_df.sort_values('amp', ascending=False).groupby('frame').first()
                       for sub_df in fiducials_dfs]
    return clean_fiducials


def find_fiducials(df, yx_shape, subsampling=1, diagnostics=False, sigmas=None, threshold=0, blob_thresh=None, **kwargs):
    """Find fiducials in pointilist PALM data
    
    The key here is to realize that there should be on fiducial per frame"""
    # incase we subsample the frame number
    num_frames = df.frame.max() - df.frame.min()
    hist_2d = palm_hist(df, yx_shape, subsampling)
    pf = PeakFinder(hist_2d, 1)
    pf.blob_sigma = 1/subsampling
    # no blobs found so try again with a lower threshold
    pf.thresh = 0
    pf.find_blobs()
    blob_thresh = max(threshold_otsu(pf.blobs[:, 3]), num_frames / 10)
    if not pf.blobs.size:
        # still no blobs then raise error
        raise RuntimeError("No blobs found!")
    pf.blobs = pf.blobs[pf.blobs[:,3] > blob_thresh]
    if pf.blobs[:, 3].max() < num_frames * subsampling / 2:
        print("Warning, drift maybe too high to find fiducials")
    # correct positions for subsampling
    pf.blobs[:, :2] = pf.blobs[:, :2] * subsampling
    return pf