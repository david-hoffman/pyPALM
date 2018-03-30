#!/usr/bin/env python
# -*- coding: utf-8 -*-
# drift.py
"""
All code related to drift correction of PALM data

Copyright (c) 2017, David Hoffman
"""

import io
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from peaks.peakfinder import PeakFinder
from skimage.filters import threshold_otsu
from .render import palm_hist
from dphutils import slice_maker

import logging
logger = logging.getLogger(__name__)

coords = ["z0", "y0", "x0"]


def find_fiducials(df, yx_shape, subsampling=1, diagnostics=False, sigmas=None, threshold=0, blob_thresh=None, **kwargs):
    """Find fiducials in pointilist PALM data
    
    The key here is to realize that there should be on fiducial per frame"""
    # incase we subsample the frame number
    num_frames = df.frame.max() - df.frame.min()
    hist_2d = palm_hist(df, yx_shape, subsampling)
    pf = PeakFinder(hist_2d.astype(int), 1 / subsampling)
    # no blobs found so try again with a lower threshold
    pf.thresh = threshold
    bkwargs = dict()
    if sigmas is not None:
        try:
            # see if the user passed more than one value
            smin, smax = sigmas
            # flip them if necessary
            if smin > smax:
                smin, smax = smax, smin
            bkwargs = dict(min_sigma=smin, max_sigma=smax)
        except TypeError:
            # only one value
            pf.blob_sigma = sigmas
    pf.find_blobs(**bkwargs)
    pf.prune_blobs(10 / subsampling)
    # need to recalculate the "amplitude" in a more inteligent way for
    # these types of data, in this case we want to take the sum over a small box
    # area
    amps = np.array([pf.data[slice_maker((int(y), int(x)), (max(1, int(s * 5)),) * 2)].sum() for y, x, s, a in pf.blobs])
    pf._blobs[:, 3] = amps
    if blob_thresh is None:
        blob_thresh = max(threshold_otsu(pf.blobs[:, 3]), num_frames / 10 * subsampling)
    pf.blobs = pf.blobs[pf.blobs[:, 3] > blob_thresh]
    if diagnostics:
        pf.plot_blobs(**kwargs)
        pf.plot_blob_grid(window=int(7 / subsampling), **kwargs)
    if not pf.blobs.size:
        # still no blobs then raise error
        raise RuntimeError("No blobs found!")
    if pf.blobs[:, 3].max() < num_frames * subsampling / 2:
        logger.warn("Drift maybe too high to find fiducials")
    # correct positions for subsampling
    return pf.blobs[:, :2] * subsampling


def extract_fiducials(df, blobs, radius, diagnostics=False):
    """Do the actual filtering
    
    We're doing it sequentially because we may run out of memory.
    If initial DataFrame is 18 GB (1 GB per column) and we have 200 """
    if diagnostics:
        pipe = None
    else:
        pipe = io.StringIO()
    fiducials_dfs = [df[np.sqrt((df.x0 - x) ** 2 + (df.y0 - y) ** 2) < radius]
                     for y, x in tqdm.tqdm(blobs, leave=False, desc="Extracting Fiducials", file=pipe)]
    return fiducials_dfs


def clean_fiducials(fiducials_dfs, order="amp", ascending=False, radius=None, zradius=None):
    """Clean up fiducials after an inital round of `extract_fiducials`

    will choose the fiducial with the largest (ascending=False), or smallest (ascending=True)
    value of `order`

    If radius is specified the fiducials will be filtered around the mean x, y position in a circle
    of radius radius."""
    if radius is not None:
        # not using z to our advantage here ...
        # the main use for this section is to clean up outliers after an initial pass with extract_fiducials
        c = ["x0", "y0"]
        r = (radius, radius)
        if zradius is not None:
            c = c + ["z0"]
            r = r + (zradius, )
        fiducials_dfs = [df[np.sqrt((((df[c] - df[c].median()) / r)**2).sum(1)) < 1]
                         for df in fiducials_dfs]
    # order fiducials in each frame by the chosen value and direction, and take the first value
    # i.e. take smallest or largest
    clean_fiducials = [sub_df.sort_values(order, ascending=ascending).groupby('frame').first()
                       for sub_df in fiducials_dfs if len(sub_df)]
    return clean_fiducials


def calc_fiducial_stats(fid_df_list, diagnostics=False, yx_pix_size=130, z_pix_size=1):
    """Calculate various stats"""
    
    def fwhm(x):
        """Return the FWHM of an assumed normal distribution"""
        return x.std() * (2 * np.sqrt(2 * np.log(2)))
    
    # keep coordinates and amplitude
    fid_stats = pd.DataFrame([f[coords + ["amp"]].mean() for f in fid_df_list])
    fid_stats[[c[0] + "drift" for c in coords]] = pd.DataFrame([f.agg({c: fwhm for c in coords}) for
                                                                f in fid_df_list])[coords]
    fid_stats["sigma"] = np.sqrt(fid_stats.ydrift**2 + fid_stats.xdrift**2)
    all_drift = pd.concat([f[coords] - f[coords].mean() for f in fid_df_list])
    if diagnostics:
        fid2plot = fid_stats[["x0", "xdrift", "y0", "ydrift", "sigma"]] * yx_pix_size
        fid2plot = pd.concat((fid2plot, fid_stats[["z0", "zdrift"]] * z_pix_size), 1)
        drift2plot = all_drift[coords] * (z_pix_size, yx_pix_size, yx_pix_size)
        fid2plot.sort_values("sigma").reset_index().plot(subplots=True)
        fid2plot.hist(bins=32)
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].get_shared_x_axes().join(axs[0], axs[1])
        for ax, k in zip(axs, ("x0", "y0", "z0")):
            d = drift2plot[k]
            fwhm = d.std() * 2 * np.sqrt(2 * np.log(2))
            bins = np.linspace(-1, 1, 64) * 2 * fwhm
            d.hist(ax=ax, bins=bins, density=True)
            ax.set_title("$\Delta {{{}}}$ = {:.0f}".format(k[0], fwhm))
            ax.set_yticks([])
        axs[1].set_xlabel("Drift (nm)")
    return fid_stats, all_drift


def remove_xy_mean(df):
    df_new = df.astype(np.float)
    xyz_mean = df_new[coords].mean()
    df_new[coords] -= xyz_mean
    return df_new.dropna()


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
            fig, axs = plt.subplots(3)
            for ff in mean_removed:
                for coord, ax in zip(coords, axs.ravel()):
                    ff[coord].plot(ax=ax)

        # want to do a weighted average
        # need to reset_index after concatination so that all localzations have unique ID
        # this will make weighting easier down the line.
        df_means = pd.concat(mean_removed).dropna().reset_index()

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

    if diagnostics:
        toreturn.plot(subplots=True)
    if frames_index is None:
        return toreturn
    else:
        assert frames_index.name == "frame"
        return toreturn.reindex(frames_index).interpolate(limit_direction="both")


def remove_drift(df_data, drift):
    """Remove the given drift from the data

    Assumes that drift is a dataframe of coordinates indexed by frame."""
    # make our index frame number so that when we subtract drift it aligns automatically along
    # the index, this needs to be tested.
    # this also, conveniently, makes a copy of the data
    df_data_dc = df_data.set_index("frame", append=True)
    # subtract drift only (assumes that drift only has these keys)
    df_data_dc[coords] -= drift
    # return the data frame with the index reset so that all localizations have
    # a unique id
    # df_data_dc.reset_index("frame", inplace=True)
    # return df_data_dc
    return df_data_dc.reset_index("frame")


def choose_good_fids(fids, max_thresh=0.25, min_thresh=0.1, min_num=5, diagnostics=False, **kwargs):
    """A heuristic to choose "good" fiducials based on their residual drift
    
    min_thresh and max_thresh are expressed in pixels"""
    # remove the drift, check residual drift, use that
    # calc the sigma for the set of fiducials
    temp_drift = calc_drift(fids, diagnostics=diagnostics)
    # remove drift from fiducials
    fids_dc = [remove_drift(fid.reset_index(), temp_drift) for fid in fids]
    s = calc_fiducial_stats(fids_dc, diagnostics=diagnostics, **kwargs)[0]
    # remove zdrift outliers
    s = s[s.zdrift <= s.zdrift.quantile(0.75)].sigma
    # sort from smallest to largest
    s = s.sort_values()
    # we have two thresholds
    # ok fiducials
    below_maxthresh = (s < max_thresh)
    # really good fiducials (wouldn't it be nice to automatically determine these thresholds from the data ... )
    below_minthresh = (s < min_thresh)
    if below_minthresh.sum() >= min_num:
        # if there's more than 5 really good fiducials, use them
        logger.debug("using only minthresh fids {}".format(min_thresh))
        good_fids = s[below_minthresh]
    else:
        if below_maxthresh.sum() == 0:
            # there's no matching fiducials, take the best one
            logger.debug("Only one good fiducial be aware")
            good_fids = s.iloc[:1]
        else:
            logger.debug("using min and max thresh = {}, {}".format(min_thresh, max_thresh))
            # use all the really good ones, plus at most five of the ok ones
            logger.debug("# below min {}  below max {}".format(below_minthresh.sum(), below_maxthresh.sum()))
            good_fids = s[below_maxthresh].iloc[:below_minthresh.sum() + min_num]
    # extract from the original list and return the new list
    logger.debug("# fids {}".format(len(good_fids)))
    return [fids[i] for i in good_fids.index], good_fids.quantile(0.75)


def weighted_avg(df, cols=coords, weight="amp"):
    # weight by inverse of sigmas of localizations
    df = df.dropna()
    temp_mean = df[cols].mul(df[weight], "index")
    temp_var = (df[cols] ** 2).mul(df[weight], "index")
    # calc weighted average
    temp_mean = temp_mean.sum().div(df[weight].sum(), "index")
    temp_std = np.sqrt(temp_var.sum().div(df[weight].sum(), "index") - temp_mean ** 2)
    temp_std.index = ["sigma_" + c[0] for c in temp_std.index]
    result = pd.concat((temp_mean, temp_std))
    result["num_counts"] = len(df)
    return result
