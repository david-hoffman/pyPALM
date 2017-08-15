# # PALM Blinking and Decay Analysis
# The purpose of this notebook is to analyze PALM diagnostic data in a consistent way across data sets. 
# The basic experiment being analyzed here is data that has been reactivated and deactivated multiple times.

import gc
import numpy as np
import pandas as pd
# regular plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# data loading
from scipy.io import readsav
from skimage.external import tifffile as tif

# get multiprocessing support
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing

# need to be able to remove fiducials
from peaks.peakfinder import PeakFinder
import tqdm

# Need partial
from functools import partial


# Register ProgressBar
ProgressBar().register()

# Lazy functions for raw data handling
lazy_imread = dask.delayed(tif.imread, pure=True)


def make_lazy_data(paths):
    """Make a lazy data array from a set of paths to data

    Assumes all data is of same shape and type."""
    lazy_data = [lazy_imread(path) for path in paths]
    # read first image for shape
    sample = tif.imread(paths[0])
    data = [dask.array.from_delayed(ld, shape=sample.shape, dtype=sample.dtype) for ld in lazy_data]
    data_array = dask.array.concatenate(data)
    return data_array


def lazy2mean(lazy_data):
    """return the mean of lazy_data"""
    return lazy_data.mean((1, 2)).compute(get=dask.multiprocessing.get)


def lazy_bg(lazy_data, num_frames=100):
    """Take the median of the last num_frames and compute the mean of the median
    frame to estimate the background and bead contributions"""
    return lazy_data[-num_frames:].median(0).mean().compute()


def peakselector_df(path, verbose=False):
    """Read a peakselector file into a pandas dataframe"""
    print("Reading {} into memory ... ".format(path))
    sav = readsav(path, verbose=verbose)
    # pull out cgroupparams, set the byteorder to native and set the rownames
    df = pd.DataFrame(sav["cgroupparams"].byteswap().newbyteorder(), columns=sav["rownames"].astype(str))
    return df


def print_maxmin(k, df):
    print("{:=^60}".format("> {} <".format(k)))
    print("{:-^60}".format("> Max <"))
    print(df.max())
    print("{:-^60}".format("> Min <"))
    print(df.min())


def grouped_peaks(df):
    """Return a DataFrame with only grouped peaks."""
    return df[df["Frame Index in Grp"] == 1]


def calc_bins(raw_data, subsampling=10):
    """Calculate the right bins from the raw data given a subsampling factor"""
    nz, ny, nx = raw_data.shape
    sub_sample_xy = 1 / subsampling
    xbins = np.arange(0, nx + sub_sample_xy, sub_sample_xy)
    ybins = np.arange(0, ny + sub_sample_xy, sub_sample_xy)
    return ybins, xbins


def filter_fiducials(df, blobs, radius):
    """Do the actual filtering
    
    We're doing it sequentially because we may run out of memory.
    If initial DataFrame is 18 GB (1 GB per column) and we have 200 """
    blob_filt = pd.Series(np.ones(len(df)), index=df.index, dtype=bool)
    for i, (y, x) in enumerate(tqdm.tqdm_notebook(blobs, leave=False, desc="Filtering Fiducials")):
        bead_filter = np.sqrt((df.xpos - x) ** 2 + (df.ypos - y) ** 2) > radius
        blob_filt &= bead_filter
        del bead_filter
        if not i % 10:
            gc.collect()
    gc.collect()
    df = df[blob_filt]
    return df


def find_fiducials(df, ybins, xbins, thresh=50, sigma=4, subsampling=10, diagnostics=False):
    """Find fiducials in pointilist PALM data"""
    # ungrouped 2d histogram to find peaks, ungrouped to make beads really stand out
    hist_2d = np.histogramdd(df[["ypos", "xpos"]].values, bins=(ybins, xbins))[0]
    pf = PeakFinder(hist_2d, sigma * subsampling)
    pf.thresh = thresh / subsampling
    sub_sample_xy = ybins[1] - ybins[0]
    # find blobs
    pf.find_blobs()
    pf.remove_edge_blobs(sigma)
    if diagnostics:
        pf._blobs[:, 3] = np.arange(len(pf.blobs))
        pf.plot_blobs(size=12, norm=LogNorm())
    return pf.blobs[:, :2] / subsampling


def remove_fiducials(df, ybins, xbins, df2=None, exclusion_radius=5, **kwargs):
    """Remove fiducials by first finding them in a histogram of localizations and then
    removing all localizations with in exclusion radius of the found ones"""
    if df2 is None:
        df2 = df
    blobs = find_fiducials(df, ybins, xbins, **kwargs)
    df3 = filter_fiducials(df2, blobs, exclusion_radius)
    # plt.matshow(np.histogramdd(df3[["Y Position", "X Position"]].values, bins=(ybins, xbins))[0], vmax=10, cmap="inferno")
    return df3


def show_frame_and_mode(lazy_data, frame_num=-1):
    """Show a given frame with a histogram of values and the mode"""
    frame = v[frame_num].compute()
    fig, (ax_im, ax_ht) = plt.subplots(1, 2, figsize=(8, 4))
    ax_im.matshow(frame, vmax=300, cmap="inferno")
    ax_im.grid("off")
    ax_im.set_title(k)
    ax_ht.hist(frame.ravel(), bins=np.logspace(2, 3, 128), log=True)
    mode = np.bincount(frame[:128, -128:].ravel()).argmax()
    ax_ht.axvline(mode, c="r")
    ax_ht.set_title("Mode = {}".format(mode))
    ax_ht.set_xscale("log")
    return fig, (ax_im, ax_ht)


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class memoize(object):
    """cache the return value of a method

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    """

    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        pfunc = partial(self, obj)
        pfunc.__doc__ = self.func.__doc__
        return pfunc

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


class RawImages(object):
    """A container for lazy raw images"""
    
    def __init__(self, paths_to_raw):
        self.raw = make_lazy_data(paths_to_raw)

    def __len__(self):
        return len(self.raw)

    @property
    def shape(self):
        return self.raw.shape

    @cached_property
    def raw_mean(self):
        """return the mean of lazy_data"""
        return self.raw.mean((1, 2)).compute(get=dask.multiprocessing.get)

    @property
    def raw_sum(self):
        return self.raw_mean * np.prod(self.raw.shape[1:])

    @memoize
    def raw_bg(self, num_frames=100):
        """Take the median of the last num_frames and compute the mean of the
        median frame to estimate the background and bead contributions"""
        return np.median(self.raw[-num_frames:], 0).mean()


class PALMExperiment(object):
    """A simple class to organize our experimental data"""
    peak_col = {
        'X Position': "xpos",
        'Y Position': "ypos",
        '6 N Photons': "nphotons",
        'Frame Number': "framenum",
        'Sigma X Pos Full': "sigmax",
        'Sigma Y Pos Full': "sigmay",
        'Z Position': 'zpos',
        'Offset': 'offset',
        'Amplitude': 'amp'
    }

    group_col = {
        'Frame Number': 'framenum',
        'Group X Position': 'xpos',
        'Group Y Position': 'ypos',
        'Group Sigma X Pos': 'sigmax',
        'Group Sigma Y Pos': 'sigmay',
        'Group N Photons': 'nphotons',
        '24 Group Size': 'groupsize',
        'Group Z Position': 'zpos',
        'Offset': 'offset',
        'Amplitude': 'amp'
    }

    def __init__(self, raw_or_paths_to_raw, path_to_sav, *args, verbose=True, init=False, **kwargs):
        """To initialize the experiment we need to know where the raw data is
        and where the peakselector processed data is
        
        Assumes paths_to_raw are properly sorted"""
        
        # deal with raw data
        if isinstance(raw_or_paths_to_raw, RawImages):
            self.raw = raw_or_paths_to_raw
        else:
            self.raw = RawImages(raw_or_paths_to_raw)
            
        # load peakselector data
        raw_df = peakselector_df(path_to_sav, verbose=verbose)
        # convert Frame number to int
        raw_df['Frame Number'] = raw_df['Frame Number'].astype(int)
        self.processed = raw_df[list(self.peak_col.keys())]
        self.grouped = grouped_peaks(raw_df)[list(self.group_col.keys())]
        # normalize column names
        self.processed = self.processed.rename(columns=self.peak_col)
        self.grouped = self.grouped.rename(columns=self.group_col)
        # initialize filtered ones
        self.processed_filtered = None
        self.grouped_filtered = None
        # do a bunch of calculations now
        if init:
            self.filter_peaks_and_beads()
            self.raw_mean

    def filter_peaks(self, offset=1000, sigma_max=3, nphotons=0, groupsize=5000):
        """Filter internal dataframes"""
        for df_title in ("processed", "grouped"):
            df = self.__dict__[df_title]
            filter_series = (
                (df.offset > 0) & # we know that offset should be around this value.
                (df.offset < offset) &
                (df.sigmax < sigma_max) &
                (df.sigmay < sigma_max) &
                (df.nphotons > nphotons)
            )
            if "groupsize" in df.keys():
                filter_series &= df.groupsize < groupsize
            self.__dict__[df_title + "_filtered"] = df[filter_series]

    def filter_peaks_and_beads(self, *args, **kwargs):
        """Filter individual localizations and remove fiducials"""
        self.filter_peaks(*args)
        self.remove_fiducials(**kwargs)

    def make_frame_report(self, **kwargs):
        return make_report(self, "Frame", **kwargs)

    def make_group_report(self, **kwargs):
        return make_report(self, "Grouped", **kwargs)

    def remove_fiducials(self, subsampling=10, exclusion_radius=5, **kwargs):
        ybins, xbins = calc_bins(self.raw.raw, subsampling)
        # use processed to find fiducials.
        for df_title in ("processed_filtered", "grouped_filtered"):
            self.__dict__[df_title] = remove_fiducials(self.processed, ybins, xbins, self.__dict__[df_title],
                                  exclusion_radius=exclusion_radius, **kwargs)

    @property
    def raw_mean(self):
        """return the mean of lazy_data"""
        return self.raw.raw_mean

    @property
    def raw_sum(self):
        return self.raw.raw_sum

    def raw_bg(self, num_frames=100):
        return self.raw.raw_bg(num_frames)


def add_line(ax, data, func=np.mean, c="k", **kwargs):
    m = func(data)
    func_name = func.__name__.capitalize()
    ax.axvline(m, color=c, label="{} = {:.3f}".format(func_name, m), **kwargs)


def log_bins(data, nbins=128):
    minmax = np.nanmin(data), np.nanmax(data)
    logminmax = np.log10(minmax)
    return np.logspace(*logminmax, num=nbins)


def make_report(expt, prefix="Frame", start=None, end=None, bins=None, nbins=None):
    """Make a nice report for Frame peaks"""
    # fixed parameters, may make arguments
    num_rep = 3

    # choose right data
    if prefix == "Frame":
        df_frame = expt.processed_filtered
        prefix = "Raw"
    elif prefix == "Grouped":
        df_frame = expt.grouped_filtered
    else:
        raise ValueError("Unrecognized Prefix")

    # get data lengths
    data_len = len(expt.raw.raw)
    rep_len = data_len // num_rep
    x_frames = np.arange(rep_len)

    # fix start and end for later filtering
    if start is None:
        start = 0
    if end is None:
        end = rep_len

    # make slice
    s = slice(start, end)
    x_frames = x_frames[s]

    # add a column which is the frame number in replicates
    df_frame["framenum_rep"] = df_frame.framenum % rep_len

    # filter what frames to look at with in each repetition
    df_frame = df_frame[(start < df_frame.framenum_rep) & (df_frame.framenum_rep < end)]

    # group the replicants together
    df_groupby_reps = df_frame.groupby(pd.cut(df_frame.framenum, np.arange(0, data_len + 1, rep_len)))
    df_groupby_frame = df_frame.groupby("framenum")

    # estimate normalization factors (proportional to total intensity and
    # therefore number of fluorophores)
    norm_factors = expt.raw_mean.reshape(num_rep, -1)[:, 0] - expt.raw_bg()

    # calculate number of photons emitted per frame and reindex to full number
    # of frames
    nphotons_per_frame = df_groupby_frame.mean().nphotons
    nphotons_per_frame = nphotons_per_frame.reindex(pd.RangeIndex(0, len(expt.raw.raw)))

    # calculate number of localizations per frame and reindex to full number
    # of frames
    localizations_per_frame = df_groupby_frame.count().nphotons
    localizations_per_frame = localizations_per_frame.reindex(pd.RangeIndex(0, len(expt.raw.raw)))

    # make data
    nphotons_per_frame = nphotons_per_frame.values.reshape(num_rep, -1)
    nphotons_norm = nphotons_per_frame / norm_factors[:, None]
    localizations_per_frame = localizations_per_frame.values.reshape(num_rep, -1)
    localizations_norm = (localizations_per_frame / norm_factors[:, None])

    vars_to_save = ["nphotons", "zpos", "ypos", "xpos", "sigmay", "sigmax", "amp", "offset"]
    return_data = dict(
        df_frame=df_frame[vars_to_save].mean(),
        df_groupby_reps=df_groupby_reps[vars_to_save].mean(),
        nphotons_per_frame=nphotons_per_frame,
        nphotons_norm=nphotons_norm,
        localizations_per_frame=localizations_per_frame,
        localizations_norm=localizations_norm
    )
    ####################################################

    # make bins
    if bins is None and nbins is None:
        # old default behaviour
        b0 = np.logspace(2.6, 6, 64)
        b1 = np.logspace(-1, 2, 64) 
        b2 = np.logspace(0, 3, 64)
        b3 = np.logspace(-2, 0.5, 64)
    elif bins is None and nbins is not None:
        b0, b1, b2, b3 = [log_bins(d, nbins)
                            for d in (
                                df_frame.nphotons.values,
                                nphotons_norm,
                                localizations_per_frame,
                                localizations_norm
                            )]
    
    if nbins is None:
        nbins = 64
        
    if isinstance(bins, tuple):
        b0, b1, b2, b3 = [np.logspace(*b, nbins) for b in bins]
    
    nphotons_bins = b0
    nphotons_norm_bins = b1
    localizations_per_frame_bins = b2
    localizations_norm_bins = b3
    
    # set up figure and axs
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    ax_gnph, hist_gnph, ax_gnph_norm, hist_gnph_norm = axs[0, :]
    ax_count, hist_count, ax_count_norm, hist_count_norm = axs[1, :]
    hist_gsx, hist_gsy, ax_decay, ax_bleach = axs[2, :]
    # plot raw numbers
    for df_fn, ax, suffix in ((nphotons_per_frame, ax_gnph, " N Photons per Emitter per Frame"),
                              (localizations_per_frame, ax_count, " Localizations per Frame")):
        ax.plot(x_frames, df_fn.T[s], alpha=0.5)
        ax.set_title(prefix + suffix)
        
    # plot normalized numbers
    for df_fn, ax, suffix in ((nphotons_norm, ax_gnph_norm, " N Photons per Emitter per Frame"),
                              (localizations_norm, ax_count_norm, " Localizations per Frame")):
        ax.plot(x_frames, df_fn.T[s], alpha=0.5)
        ax.set_title("Normalized " + prefix + suffix)

    # Label histograms
    hist_gnph.set_title(prefix + " N Photons")
    hist_gsx.set_title(prefix + " Sigma X")
    hist_gsy.set_title(prefix + " Sigma Y")
    
    # plot number of photons histogram
    df_groupby_reps.nphotons.hist(bins=nphotons_bins, log=True, ax=hist_gnph, alpha=0.5, normed=True)
    ## reset color cycle for lines
    hist_gnph.set_prop_cycle(None)
    df_groupby_reps.nphotons.hist(bins=nphotons_bins, log=True, ax=hist_gnph, histtype="step", linewidth=2, normed=True)
    add_line(hist_gnph, df_frame.nphotons, ls=":")
    add_line(hist_gnph, df_frame.nphotons, np.median, ls="--")
    hist_gnph.legend()
    hist_gnph.set_xscale("log")

    
    for col, ax in zip(("sigmax", "sigmay"), (hist_gsx, hist_gsy)):
        bins=np.linspace(0, 1, nbins)
        df_groupby_reps[col].hist(
            bins=bins,
            log=False,
            alpha=0.5,
            ax=ax,
            normed=True
        )
        ax.set_prop_cycle(None)
        df_groupby_reps[col].hist(
            bins=bins,
            log=False,
            histtype="step",
            linewidth=2,
            ax=ax,
            normed=True
        )
        add_line(ax, df_frame[col], ls=":")
        add_line(ax, df_frame[col], np.median, ls="--")
        ax.legend()

    
    for ax, dd, bins, suffix in zip((hist_gnph_norm, hist_count, hist_count_norm),
                                    (nphotons_norm, localizations_per_frame, localizations_norm),
                                    (nphotons_norm_bins, localizations_per_frame_bins, localizations_norm_bins),
                                    (" N Photons Norm", " Localizations", " Localizations Norm")):
        for d in dd:
            d = d[np.isfinite(d)]
            ax.hist(d, bins=bins, log=True, alpha=0.5, normed=True)
        ax.set_prop_cycle(None)
        for d in dd:
            d = d[np.isfinite(d)]
            ax.hist(d, bins=bins, log=True, histtype="step", linewidth=2, normed=True)

        ax.set_title(prefix + suffix)
        add_line(ax, d, ls=":")
        add_line(ax, d, np.median, ls="--")
        ax.legend()
        ax.set_xscale("log")

    
    ax_bleach.set_title("Bleaching")
    ax_bleach.plot(norm_factors / norm_factors.max(), "-o", markerfacecolor='red', markeredgecolor='k')
    
    ax_decay.set_title("Intensity Decays")
    ax_decay.plot(x_frames, ((expt.raw_mean.reshape(3, -1) - expt.raw_bg())/norm_factors[:, None]).T[s], alpha=0.5)
    
    for ax in (ax_decay, ax_gnph, ax_gnph_norm, ax_count, ax_count_norm):
        ax.set_xlim(0, rep_len)
    
    fig.tight_layout()
    
    return fig, axs, return_data
