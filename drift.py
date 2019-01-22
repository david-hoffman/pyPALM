#!/usr/bin/env python
# -*- coding: utf-8 -*-
# drift.py
"""
All code related to drift correction of PALM data

Copyright (c) 2017, David Hoffman
"""

import io
import gc
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
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
    logger.debug("num_frames = {}".format(num_frames))
    hist_2d = palm_hist(df, yx_shape, subsampling)
    logger.debug("subsampling = {}".format(subsampling))
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
    logger.debug("bkwargs = {}".format(bkwargs))
    pf.find_blobs(**bkwargs)
    # need to recalculate the "amplitude" in a more inteligent way for
    # these types of data, in this case we want to take the sum over a small box
    # area
    blobs = pf.blobs
    amps = np.array([pf.data[slice_maker((int(y), int(x)), (max(1, int(s * 5)),) * 2)].sum() for y, x, s, a in blobs])
    logger.debug("amps = {}".format(amps))
    blobs[:, 3] = amps
    if blob_thresh is None:
        blob_thresh = max(threshold_otsu(blobs[:, 3]), num_frames / 10 * subsampling)
    logger.debug("blob_thresh = {}".format(blob_thresh))
    pf.blobs = blobs[blobs[:, 3] > blob_thresh]
    if not pf.blobs.size:
        # still no blobs then raise error
        raise RuntimeError("No blobs found!")
    if diagnostics:
        pf.plot_blobs(**kwargs)
        pf.plot_blob_grid(window=int(7 / subsampling), **kwargs)
    if pf.blobs[:, 3].max() < num_frames * subsampling / 2:
        logger.warn("Drift maybe too high to find fiducials localizations = {}, num_frames = {}".format(pf.blobs[:, 3].max(), num_frames))
    # correct positions for subsampling
    fid_locs = pf.blobs[:, :2] * subsampling
    logger.debug("fid_locs = {}".format(fid_locs))
    return fid_locs


def extract_fiducials(df, blobs, radius, diagnostics=False):
    """Do the actual filtering
    
    We're doing it sequentially because we may run out of memory.
    If initial DataFrame is 18 GB (1 GB per column) and we have 200 """
    if diagnostics:
        # turn blobs into tqdm generator instead
        blobs = tqdm.tqdm_notebook(blobs, desc="Extracting Fiducials")
    fiducials_dfs = [df[np.sqrt((df.x0 - x) ** 2 + (df.y0 - y) ** 2) < radius]
                     for y, x in blobs]
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


def calc_residual_drift(fid_df_list):
    return pd.concat([fid[coords] - fid[coords].mean() for fid in fid_df_list], ignore_index=True).std()


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
        fid2plot.sort_values("sigma").reset_index().plot(subplots=True, figsize=(4, 8))
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


def remove_xyz_mean(df):
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
        toreturn = remove_xyz_mean(fiducials_df[0])[coords]
    else:
        mean_removed = [remove_xyz_mean(ff) for ff in fiducials_df]
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
        if weighted.lower() == "coords":
            # save the labels for weighted coords and weights
            w_coords = []
            weights = []
            # loop through coords generating weights and weighted coords
            for x in coords:
                c = x[0]
                s = "sigma_" + c
                df_means[s + "_inv"] = 1 / df_means[s] ** 2
                weights.append(s + "_inv")
                df_means[x + "_w"] = df_means[x].mul(df_means[s + "_inv"], "index")
                w_coords.append(x + "_w")
            # groupby group_id and sum
            temp_gb = df_means.groupby("frame")
            # finish weighted mean
            new_coords = temp_gb[w_coords].sum() / temp_gb[weights].sum().values
            new_coords.columns = [c.replace("_w", "") for c in new_coords.columns]
            # calc new sigma
            # return new data frame
            toreturn = new_coords
        elif weighted:
            # weight the coordinates
            logger.debug("Weighting by {}".format(weighted))
            df_means[coords] = df_means[coords].mul(df_means[weighted], "index")
            # groupby frame
            temp = df_means.groupby("frame")
            # calc weighted average
            toreturn = temp[coords].sum().div(temp[weighted].sum(), "index")
        else:
            toreturn = df_means.groupby("frame")[coords].mean()
        # remove mean of total drift.
        toreturn = remove_xyz_mean(toreturn)
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


def choose_good_fids(fids, max_thresh=0.25, min_thresh=0.1, min_num=5, diagnostics=False, z_quantile=0.75, **kwargs):
    """A heuristic to choose "good" fiducials based on their residual drift
    
    min_thresh and max_thresh are expressed in pixels"""
    # remove the drift, check residual drift, use that
    # calc the sigma for the set of fiducials
    temp_drift = calc_drift(fids, diagnostics=diagnostics)
    # remove drift from fiducials
    fids_dc = [remove_drift(fid.reset_index(), temp_drift) for fid in fids]
    s = calc_fiducial_stats(fids_dc, diagnostics=diagnostics, **kwargs)[0]
    # remove zdrift outliers
    s = s[s.zdrift <= s.zdrift.quantile(z_quantile)].sigma
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
    return [fids[i] for i in good_fids.index], good_fids.quantile(z_quantile)


def remove_all_drift(data, yx_shape, init_drift, frames_index, atol=1e-6, rtol=1e-3, maxiters=100,
                     max_thresh=0.5, min_thresh=0.25, min_num=8, clean=True, diagnostics=False,
                     capture_radius=None, max_extraction=20, weighted="coords", order="sigma_z",
                     ascending=True, init_iters=0, **kwargs):
    """Iteratively remove drift from the data

    Parameters
    ----------

    Returns
    -------
    data_dc, init_drift, delta_drift, good_fids_dc
    """
    # make a copy of the initial drift so it doesn't get overwritten

    # make a simple function

    def find_fiducials_simple(data_dc, capture_radius):
        return find_fiducials(data_dc, yx_shape, subsampling=1,
                              diagnostics=diagnostics, cmap="inferno", norm=PowerNorm(0.25),
                              sigmas=(1 / np.sqrt(1.6), max(capture_radius, np.sqrt(1.6) * 0.99)),
                              **kwargs)

    if init_drift is None:
        if frames_index is None:
            temp_frames = pd.RangeIndex(data.frame.min(), data.frame.max() + 1, name="frame")
        else:
            temp_frames = frames_index
        init_drift = pd.DataFrame(0, index=temp_frames, columns=coords)
        if capture_radius is None:
            fid0 = find_fiducials_simple(data, 50)[:1]
            fid0 = extract_fiducials(data, fid0, 50)[0]
            capture_radius = max(fid0.std()[["x0", "y0"]].mean(), 5)
    else:
        init_drift = init_drift.copy()
        # calculate the inital capture radius for gathering fiducials
        if capture_radius is None:
            capture_radius = min(50, np.abs(init_drift[["x0", "y0"]].values).max())
    # initialize delta_drift and old_drift to nan so that one round of iteration occurs
    delta_drift = init_drift.iloc[:1] * np.nan
    old_drift = np.nan
    # begin iteration
    for i in range(maxiters):
        # If the user requests it make sure that the drift is interpolated.
        # If the drift isn't interpolated then frames without fiducials will be dropped.
        if frames_index is not None:
            init_drift = init_drift.reindex(frames_index).interpolate("slinear", fill_value="extrapolate", limit_direction="both")

        logger.info("capture_radius {:.3f}".format(capture_radius))

        # remove drift
        if init_drift.iloc[0].sum() != 0:
            data_dc = remove_drift(data, init_drift).dropna()
        else:
            data_dc = data

        # calculate the remaining drift as the RMS of the residual drift
        avg_drift = np.sqrt((delta_drift[["x0", "y0"]].std()**2).sum(skipna=False))

        logger.info("{}: drift {:.2e}".format(i, avg_drift))

        # check if the drift is below atol or hasn't changed that much (rtol)
        if avg_drift <= atol or abs(old_drift - avg_drift) / old_drift < rtol:
            break

        # save the avg_drift
        old_drift = avg_drift

        # find fiducials
        fids_locations_dc = find_fiducials_simple(data_dc, capture_radius)
        # extract fiducials
        fids_dc = extract_fiducials(data_dc, fids_locations_dc[:max_extraction], max(capture_radius, 1), diagnostics=diagnostics)
        # filter fids based on extent
        if clean:
            # pick smallest sigma_z
            if capture_radius < 1:
                radius = capture_radius
            else:
                radius = None
            fids_dc = clean_fiducials(fids_dc, order=order, ascending=ascending, radius=radius)

        # choose "good" fiducials
        good_fids_dc, s_max = choose_good_fids(fids_dc, max_thresh=max_thresh,
                                               min_thresh=min_thresh,
                                               min_num=min_num,
                                               diagnostics=diagnostics, z_quantile=0.99)

        if capture_radius > 3:
            # early on no good fids
            logger.info("Drift large using min_num fiducials")
            good_fids_dc = fids_dc[:min_num]

        logger.info("max_s = {:.3f}".format(s_max))
        # update capture radius if not in init iterations.
        if i >= init_iters - 1:
            capture_radius = max(s_max * 3, 0.5)
        # calculate the delta drift
        delta_drift = calc_drift(good_fids_dc, weighted=weighted, frames_index=frames_index,
                                 diagnostics=diagnostics)
        # if there's only one fiducial use a rolling median for the drift.
        if len(good_fids_dc) < 2:
            logger.warn("Only one fiducial, smoothing drift")
            delta_drift = delta_drift.rolling(100, 0, center=True, win_type="gaussian").mean(std=20)
            capture_radius = max(np.abs(delta_drift[["x0", "y0"]].values).max(), 1)
        # update total drift
        init_drift += delta_drift
        gc.collect()
    else:
        logger.warn("Reached maxiters {}".format(maxiters))

    if diagnostics:
        print(len(good_fids_dc))
        calc_fiducial_stats(good_fids_dc, diagnostics=diagnostics)
        plt.show()

    return data_dc, init_drift, delta_drift, good_fids_dc
