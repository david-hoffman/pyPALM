#!/usr/bin/env python
# -*- coding: utf-8 -*-
# grouping.py
"""
All code related to drift correction of PALM data

Copyright (c) 2017, David Hoffman
"""
import os
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import dask
import tempfile
# get a logger
import logging
logger = logging.getLogger(__name__)


def calculate_zscaling(data, diagnostics=False):
    """Calculate the zscaling of the data"""
    # look at ratio of axial extent to lateral extent of gaussian clouds
    aspect = data.sigma_z / np.sqrt((data[["sigma_x", "sigma_y"]]**2).sum(1))
    aspect = aspect[(aspect < aspect.quantile(0.999)) & (aspect > aspect.quantile(1 - 0.999))]
    median = aspect.median()
    if diagnostics:
        fig, ax = plt.subplots(figsize=(5, 4))
        aspect.hist(bins="auto", density=True, ax=ax, histtype="step", linewidth=1)
        ax.axvline(median, c="C1", label="Median Ratio = {:.1f}".format(median), linewidth=3, linestyle="--")
        ax.set_xlabel("Ratio of PSF Axial to Lateral Extent")
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.grid(False)
        ax.legend()
    return median


def estimate_grouping_radius(df, sample_size=128, boot_samples=512, zscaling=None, drift=None, quantiles=(0.9, 0.99, 0.999)):
    """Estimate the correct grouping radius from the data, assumes that `df` is the result of
    a single pass group (grouping contiguous--in time--localizations)"""
    if drift is not None:
        # copy relevant parameters and add drift in quadrature
        logger.debug("using drift")
        df = np.sqrt(df[["sigma_x", "sigma_y", "sigma_z"]]**2 + drift[["x0", "y0", "z0"]].values**2)
    else:
        df = df[["sigma_x", "sigma_y", "sigma_z"]]

    # do you want to include z in the analysis?
    if zscaling is not None:
        # if so scale sigma_z
        logger.debug("using zscaling")
        df = df[["sigma_x", "sigma_y", "sigma_z"]] / (1, 1, zscaling)
    else:
        df = df[["sigma_x", "sigma_y"]]

    # generate boot strap samples
    boot_strap = []
    for i in tqdm.tnrange(boot_samples):
        # Make fake 2D / 3D point cloud
        sample = df.sample(sample_size, replace=True) * np.random.randn(sample_size, len(df.columns))
        # calculate r for each point
        sample = np.sqrt((sample ** 2).sum(1))
        # pick quantiles of r
        boot_strap.append(sample.quantile(quantiles))

    # return the median value of all boot strapped samples
    return pd.DataFrame(boot_strap).median()


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


def group(df, radius, gap, zscaling=None, frame_reset=np.inf):
    """Group peaks based on x y locations

    Parameters
    ----------
    df : pandas DataFrame
    radius : float
    gap : int
    zscaling: float
        Scale the z0 values and include them in point matching routine
        i.e. if np.sqrt((x1 - x0)**2 + (y1 - y0)**2  + ((z1 - z0) / zcaling)**2) < r
        then group points 1 and 0.
    frame_reset : int
    """

    # define norm functions for later
    if zscaling is None:
        def norm(df_sub):
            """return y, x pairs"""
            return df_sub[["y0", "x0"]].values
    else:
        def norm(df_sub):
            """return z, y, x pairs with normalized z for point matching"""
            return df_sub[["z0", "y0", "x0"]].values / (zscaling, 1, 1)

    new_df_list = []
    # should add a progress bar here
    # cycle through all frames
    frame_min = df.frame.min()
    for frame, peaks in df.groupby("frame"):
        peaks = peaks.copy()

        # set/reset group_id
        peaks["group_id"] = -1

        # reset cache if required
        if not (frame - frame_min) % frame_reset:
            # group_id will be the index of the first peak
            df_cache = peaks
            df_cache.loc[peaks.index, "group_id"] = df_cache.index
            new_df_list.append(df_cache.copy())
            continue

        # clear cache of old stuff
        df_cache = df_cache[(frame - df_cache.frame) <= gap]

        if len(df_cache):
            # if anything is still in the cache look for matches
            # search for matches
            matches = find_matches([norm(df_cache), norm(peaks)], radius)
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
        new_df_list.append(peaks)

    return pd.concat(new_df_list)


# new code
def agg_groups(df_grouped):
    """Aggregate groups, weighted mean as usual, and sigmas are standard error on the weighted
    mean as calculated in the reference below.

    Gatz, Donald F., and Luther Smith.
    “The Standard Error of a Weighted Mean Concentration—I. Bootstrapping vs Other Methods.”
    Atmospheric Environment 29, no. 11 (June 1, 1995): 1185–93.
    https://doi.org/10.1016/1352-2310(94)00210-C
    """
    coords = ["x", "y", "z"]

    # turns out that its fastest to use pandas aggs built in functions at all
    # costs, even more memory, so we need to build the columns we'll use
    # later on

    # coordinates
    sigmas = []
    # coordinates
    xi = []
    # weights
    wi = []
    # square weights
    wi2 = []
    # weighted coordinates
    wi_xi = []
    # square weighted coordinates
    wi2_xi = []
    # square weighted squared coordinates
    wi2_xi2 = []

    # loop through coords generating weights and weighted coords
    for c in coords:
        s = "sigma_" + c
        sigmas.append(s)
        x = c + "0"
        xi.append(x)
        # calculate weights
        wi.append("wi_" + c)
        df_grouped[wi[-1]] = 1 / df_grouped[s] ** 2
        # square weights
        wi2.append("wi2_" + c)
        df_grouped[wi2[-1]] = 1 / df_grouped[s] ** 4
        # weighted position
        wi_xi.append("wi_xi_" + c)
        df_grouped[wi_xi[-1]] = df_grouped[x].mul(df_grouped[wi[-1]], "index")
        # square weighted coordinates
        wi2_xi.append("wi2_xi_" + c)
        df_grouped[wi2_xi[-1]] = df_grouped[x].mul(df_grouped[wi2[-1]], "index")
        # square weighted squared coordinates
        wi2_xi2.append("wi2_xi2_" + c)
        df_grouped[wi2_xi2[-1]] = (df_grouped[x] ** 2).mul(df_grouped[wi2[-1]], "index")

    # groupby group_id and sum
    temp_gb = df_grouped.groupby("group_id")

    # calc new group params
    new_amp = temp_gb[["amp", "nphotons", "chi2", "offset"]].sum()
    new_frame = temp_gb[["frame"]].first()
    groupsize = temp_gb.size()
    groupsize.name = "groupsize"

    # calculate sum weights
    wi_bar = temp_gb[wi].sum().values

    # finish weighted mean
    mu = temp_gb[wi_xi].sum() / wi_bar

    # doing this here to preserve order
    mu.columns = [c[-1] + "0" for c in mu.columns]

    # calc new sigma
    new_sigmas = (temp_gb[wi2_xi2].sum().values
                  - 2 * mu[xi] * temp_gb[wi2_xi].sum().values
                  + (mu[xi] ** 2) * temp_gb[wi2].sum().values)
    gsize = groupsize.values[:, None]
    # we know there'll be floating point errors from the following
    # because we'll divide by zero for groups with one point
    with np.errstate(divide='ignore', invalid='ignore'):
        new_sigmas = np.sqrt((gsize / (gsize - 1)) * new_sigmas / wi_bar ** 2)
    new_sigmas.columns = ["sigma_" + c[0] for c in xi]
    # find the places we divided by zero and replace with single localization sigma
    nan_locs = ~np.isfinite(new_sigmas).all(1)
    new_sigmas[nan_locs] = temp_gb[sigmas].first()[nan_locs]

    # take the mean of all remaining columns
    # figure out columns to drop
    extra_columns = wi + wi2 + wi_xi + wi2_xi + wi2_xi2
    other_columns = ["groupsize"] + extra_columns
    for df in (mu, new_sigmas, new_amp, new_frame):
        other_columns += df.columns.tolist()
    other_columns += ["group_id"]
    columns_to_mean = df_grouped.columns.difference(other_columns)
    # take the mean
    if len(columns_to_mean):
        new_means = temp_gb[columns_to_mean].mean()
    else:
        new_means = pd.DataFrame()

    # drop added columns from original data frame
    df_grouped.drop(columns=extra_columns, inplace=True)

    # return new data frame
    df_agg = pd.concat([mu, new_sigmas, new_amp, new_frame, groupsize, new_means], axis=1)
    return df_agg


@dask.delayed
def grouper(df, *args, **kwargs):
    """Delayed wrapper around grouping and aggregating code"""
    if len(df):
        df = agg_groups(group(df, *args, **kwargs))
    logger.info("Finished grouping {}, {}".format(args, kwargs))
    return df


@dask.delayed
def _grouper_to_file(filename, df, *args, **kwargs):
    """Delayed wrapper around grouping and aggregating code"""
    if len(df):
        grouped_df = agg_groups(group(df, *args, **kwargs))
        # log
        logger.debug("Writing data to {}".format(filename))
        grouped_df.to_hdf(filename, "data")
        return filename


@dask.delayed
def _file_to_grouper(filename):
    return pd.read_hdf(filename, "data")


def chunked_grouper(df, *args, numthreads=24, concat=True, **kwargs):
    """Chunk data and delayed group, return a list"""
    length = len(df)
    chunklen = (length + numthreads - 1) // numthreads
    # Create argument tuples for each input chunk
    grouped = [grouper(df.iloc[i * chunklen:(i + 1) * chunklen], *args, **kwargs)
               for i in range(numthreads)]
    if concat:
        # make the concatenation step a task, this can result
        # in failure if the data is too large and processes are used,
        # because the data is concatenated remotely and then brought
        # back to the main process.
        return dask.delayed(pd.concat)(grouped, ignore_index=True)
    else:
        return grouped


def _chunked_grouper_to_dir(directory, df, *args, numthreads=24, **kwargs):
    """Chunk data and delayed group, return a list"""
    length = len(df)
    chunklen = (length + numthreads - 1) // numthreads
    # Create argument tuples for each input chunk
    basename = directory + "GrpFile{:05d}.h5"
    logger.debug("Basename = {}".format(basename))
    grouped = [_grouper_to_file(basename.format(i), df.iloc[i * chunklen:(i + 1) * chunklen], *args, **kwargs)
               for i in range(numthreads)]
    return grouped


def slab_grouper(slabs, *args, **kwargs):
    """Take a list of slabs, group them and return the grouped slabs"""
    # intermediate results will be written to disk
    # do everything within a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # compute the groups, which are saved to file
        logger.debug("tmpdir = {}".format(tmpdir))
        dirname = os.path.join(tmpdir, "SlabDir{:05d}")

        logger.info("Beginning calculation ...")
        grouped_slabs_dirs = dask.delayed([
            # make sure that each slabe is chunked grouped in a temporary dir in our main tempdir
            _chunked_grouper_to_dir(dirname.format(i), slab, *args, **kwargs) for i, slab in enumerate(slabs)
        ]).compute(scheduler="processes")
        logger.info("... finishing calculation.")

        # read back data into slabs
        logger.info("Beginning reading back data ...")
        grouped_slabs = dask.delayed([dask.delayed(pd.concat)(dask.delayed([_file_to_grouper(f) for f in d]))for d in grouped_slabs_dirs])
        grouped_slabs = grouped_slabs.compute()
        logger.info("... finishing reading back data")

    return grouped_slabs


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
