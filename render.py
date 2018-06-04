#!/usr/bin/env python
# -*- coding: utf-8 -*-
# render.py
"""
All code related to visualization of PALM data

Copyright (c) 2017, David Hoffman
"""

import gc
import json
import os
import numpy as np
from numpy.core import atleast_1d, atleast_2d
from numba import njit
from dphutils import _calc_pad, scale
from dphplotting import auto_adjust

import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.colors import Normalize, PowerNorm, ListedColormap
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from skimage.external import tifffile as tif
from scipy.misc import imsave

# get multiprocessing support
import dask
import dask.array

import logging
logger = logging.getLogger()

greys_alpha_cm = ListedColormap([(i / 255,) * 3 + ((255 - i) / 255,) for i in range(256)])


def choose_dtype(max_value):
    """choose the appropriate dtype for saving images"""
    # if any type of float, use float32
    if np.issubdtype(max_value, np.inexact):
        return np.float32
    # check for integers now
    if max_value < 2**8:
        return np.uint8
    elif max_value < 2**16:
        return np.uint16
    elif max_value < 2**32:
        return np.uint32
    return np.float32


def tif_convert(data):
    """convert data for saving as tiff copying if necessary"""
    data = np.asarray(data)
    return data.astype(choose_dtype(data.max()), copy=False)


def palm_hist(df, yx_shape, subsampling=1):
    """Return a histogram of the data"""
    # the function has been refactored to use the gen_img code
    return gen_img(yx_shape, df, mag=1 / subsampling, cmap=None, hist=True)


@njit(nogil=True)
def jit_hist3d(zpositions, ypositions, xpositions, shape):
    """Generate a histogram of points in 3D

    Parameters
    ----------

    Returns
    -------
    res : ndarray
    """
    nz, ny, nx = shape
    res = np.zeros(shape, np.uint32)
    # need to add ability for arbitraty accumulation
    for z, y, x in zip(zpositions, ypositions, xpositions):
        # bounds check
        if x < nx and y < ny and z < nz:
            res[z, y, x] += 1
    return res


@njit(nogil=True)
def jit_hist3d_with_weights(zpositions, ypositions, xpositions, weights,
                            shape):
    """Generate a histogram of points in 3D

    Parameters
    ----------

    Returns
    -------
    res : ndarray
    """
    nz, ny, nx = shape
    res = np.zeros(shape, weights.dtype)
    # need to add ability for arbitraty accumulation
    for z, y, x, w in zip(zpositions, ypositions, xpositions, weights):
        # bounds check
        if x < nx and y < ny and z < nz:
            res[z, y, x] += w
    return res


def fast_hist3d(sample, bins, myrange=None, weights=None):
    """Modified from numpy histogramdd
    Make a 3d histogram, fast, lower memory footprint"""
    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = atleast_2d(sample).T
        N, D = sample.shape

    nbin = np.empty(D, int)
    edges = D * [None]
    dedges = D * [None]

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the'
                ' sample x.')
    except TypeError:
        # bins is an integer
        bins = D * [bins]

    # Select range for each dimension
    # Used only if number of bins is given.
    if myrange is None:
        # Handle empty input. Range can't be determined in that case, use 0-1.
        if N == 0:
            smin = np.zeros(D)
            smax = np.ones(D)
        else:
            smin = atleast_1d(np.array(sample.min(0), float))
            smax = atleast_1d(np.array(sample.max(0), float))
    else:
        if not np.all(np.isfinite(myrange)):
            raise ValueError(
                'myrange parameter must be finite.')
        smin = np.zeros(D)
        smax = np.zeros(D)
        for i in range(D):
            smin[i], smax[i] = myrange[i]

    # Make sure the bins have a finite width.
    for i in range(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # avoid rounding issues for comparisons when dealing with inexact types
    if np.issubdtype(sample.dtype, np.inexact):
        edge_dt = sample.dtype
    else:
        edge_dt = float
    # Create edge arrays
    for i in range(D):
        if np.isscalar(bins[i]):
            if bins[i] < 1:
                raise ValueError(
                    "Element at index %s in `bins` should be a positive "
                    "integer." % i)
            nbin[i] = bins[i] + 2  # +2 for outlier bins
            edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1, dtype=edge_dt)
        else:
            edges[i] = np.asarray(bins[i], edge_dt)
            nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
        dedges[i] = np.diff(edges[i]).min()
        if np.any(np.asarray(dedges[i]) <= 0):
            raise ValueError(
                "Found bin edge of size <= 0. Did you specify `bins` with"
                "non-monotonic sequence?")

    nbin = np.asarray(nbin)

    # Handle empty input.
    if N == 0:
        return np.zeros(nbin - 2), edges

    # Compute the bin number each sample falls into.
    # np.digitize returns an int64 array when it only needs to be uint32
    Ncount = [np.digitize(sample[:, i], edges[i]).astype(np.uint32) for i in range(D)]
    shape = tuple(len(edges[i]) - 1 for i in range(D))  # -1 for outliers

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in range(D):
        # Rounding precision
        mindiff = dedges[i]
        if not np.isinf(mindiff):
            decimal = int(-np.log10(mindiff)) + 6
            # Find which points are on the rightmost edge.
            not_smaller_than_edge = (sample[:, i] >= edges[i][-1])
            on_edge = (np.around(sample[:, i], decimal) ==
                       np.around(edges[i][-1], decimal))
            # Shift these points one bin to the left.
            Ncount[i][np.where(on_edge & not_smaller_than_edge)[0]] -= 1

    # Flattened histogram matrix (1D)
    # Reshape is used so that overlarge arrays
    # will raise an error.
    # hist = zeros(nbin, float).reshape(-1)

    # # Compute the sample indices in the flattened histogram matrix.
    # ni = nbin.argsort()
    # xy = zeros(N, int)
    # for i in arange(0, D-1):
    #     xy += Ncount[ni[i]] * nbin[ni[i+1:]].prod()
    # xy += Ncount[ni[-1]]

    # # Compute the number of repetitions in xy and assign it to the
    # # flattened histmat.
    # if len(xy) == 0:
    #     return zeros(nbin-2, int), edges

    # flatcount = bincount(xy, weights)
    # a = arange(len(flatcount))
    # hist[a] = flatcount

    # # Shape into a proper matrix
    # hist = hist.reshape(sort(nbin))
    # for i in arange(nbin.size):
    #     j = ni.argsort()[i]
    #     hist = hist.swapaxes(i, j)
    #     ni[i], ni[j] = ni[j], ni[i]

    # # Remove outliers (indices 0 and -1 for each dimension).
    # core = D*[slice(1, -1)]
    # hist = hist[core]

    # # Normalize if normed is True
    # if normed:
    #     s = hist.sum()
    #     for i in arange(D):
    #         shape = ones(D, int)
    #         shape[i] = nbin[i] - 2
    #         hist = hist / dedges[i].reshape(shape)
    #     hist /= s

    # if (hist.shape != nbin - 2).any():
    #     raise RuntimeError(
    #         "Internal Shape Error")
    # for n in Ncount:
    #     print(n.shape)
    #     print(n.dtype)
    if weights is not None:
        weights = np.asarray(weights)
        hist = jit_hist3d_with_weights(*Ncount, weights=weights, shape=shape)
    else:
        hist = jit_hist3d(*Ncount, shape=shape)
    return hist, edges

### Gaussian Rendering
_jit_calc_pad = njit(_calc_pad, nogil=True)


@njit(nogil=True)
def _jit_slice_maker(xs1, ws1):
    """Modified from the version in dphutils to allow jitting"""
    if np.any(ws1 < 0):
        raise ValueError("width cannot be negative")
    # ensure integers
    xs = np.rint(xs1).astype(np.int32)
    ws = np.rint(ws1).astype(np.int32)
    # use _calc_pad
    toreturn = []
    for x, w in zip(xs, ws):
        half2, half1 = _jit_calc_pad(0, w)
        xstart = x - half1
        xend = x + half2
        assert xstart <= xend, "xstart > xend"
        if xend <= 0:
            xstart, xend = 0, 0
        # the max calls are to make slice_maker play nice with edges.
        toreturn.append((max(0, xstart), xend))
    # return a list of slices
    return toreturn


def _gauss(yw, xw, y0, x0, sy, sx):
    """Simple normalized 2D gaussian function for rendering"""
    # for this model, x and y are seperable, so we can generate
    # two gaussians and take the outer product
    y, x = np.arange(yw), np.arange(xw)
    amp = 1 / (2 * np.pi * sy * sx)
    gy = np.exp(-((y - y0) / sy) ** 2 / 2)
    gx = np.exp(-((x - x0) / sx) ** 2 / 2)
    return amp * np.outer(gy, gx)

_jit_gauss = njit(_gauss, nogil=True)


def _gen_img_sub(yx_shape, params, mag, multipliers, diffraction_limit):
    """A sub function for actually rendering the images
    Some of the structure is not really 'pythonic' but its to allow JIT compilation

    Parameters
    ----------
    yx_shape : tuple
        The shape overwhich to render the scene
    params : ndarray (M x N)
        An array containing M localizations with data ordered as
        y0, x0, sigma_y, sigma_x
    mag : int
        The magnification factor to render the scene
    multipliers : array (M) optional
        an array of multipliers so that you can do weigthed averages
        mainly to be used for depth coded MIPs
    diffraction_limit : bool
        Controls whether or not there is a lower limit for the localization precision
        This can have better smoothing.

    Returns
    -------
    img : ndarray
        The rendered image
    """
    # hard coded radius, this is really how many sigmas you want
    # to use to render each gaussian
    radius = 5
    yw, xw = yx_shape
    # initialize the image
    img = np.zeros((yw * mag, xw * mag))
    # iterate through all localizations
    for i in range(len(params)):
        if not np.isfinite(params[i]).all():
            # skip nans
            continue
        # pull parameters
        y0, x0, sy, sx = params[i]
        # adjust to new magnification
        y0, x0, sy, sx = np.array((y0, x0, sy, sx)) * mag
        if diffraction_limit:
            sy, sx = max(sy, 0.5), max(sx, 0.5)
        # calculate the render window size
        width = np.array((sy, sx)) * radius * 2
        # calculate the area in the image
        (ystart, yend), (xstart, xend) = _jit_slice_maker(np.array((y0, x0)), width)
        # adjust coordinates to window coordinates
        y0 -= ystart
        x0 -= xstart
        # generate gaussian
        g = _jit_gauss((yend - ystart), (xend - xstart), y0, x0, sy, sx)
        # weight if requested
        if len(multipliers):
            g *= multipliers[i]
        # update image
        img[ystart:yend, xstart:xend] += g

    return img

_jit_gen_img_sub = njit(_gen_img_sub, nogil=True)


# @pdiag.dask.delayed
# def _gen_img_sub_thread(chunklen, chunk, yx_shape, df, mag, multipliers, diffraction_limit):
#     """"""
#     s = slice(chunk * chunklen, (chunk + 1) * chunklen)
#     df_chunk = df[["y0", "x0", "sigma_y", "sigma_x"]].values[s]
#     # calculate the amplitude of the z gaussian.
#     amps = multipliers[s]
#     # generate a 2D image weighted by the z gaussian.
#     return pdiag._jit_gen_img_sub(yx_shape, df_chunk, mag, amps, diffraction_limit)


# def _gen_img_sub_threaded(yx_shape, df, mag, multipliers, diffraction_limit, numthreads=1):
#     """"""
#     length = len(df)
#     chunklen = (length + numthreads - 1) // numthreads
#     new_shape = tuple(np.array(yx_shape) * mag)
#     # print(dask.array.from_delayed(_gen_zplane(df, yx_shape, zplanes[0], mag), new_shape, np.float))
#     rendered_threads = [pdiag.dask.array.from_delayed(
#         _gen_img_sub_thread(chunklen, chunk, yx_shape, df, mag, multipliers, diffraction_limit), new_shape, np.float)
#                         for chunk in range(numthreads)]
#     lazy_result = pdiag.dask.array.stack(rendered_threads)
#     return lazy_result.sum(0)

def _gen_img_sub_threaded(yx_shape, df, mag, multipliers, diffraction_limit, numthreads=1):
    keys_for_render = ["y0", "x0", "sigma_y", "sigma_x"]
    df = df[keys_for_render].values
    length = len(df)
    chunklen = (length + numthreads - 1) // numthreads
    # Create argument tuples for each input chunk
    df_chunks = [df[i * chunklen:(i + 1) * chunklen] for i in range(numthreads)]
    mult_chunks = [multipliers[i * chunklen:(i + 1) * chunklen] for i in range(numthreads)]
    delayed_jit_gen_img_sub = dask.delayed(_jit_gen_img_sub)
    lazy_result = [delayed_jit_gen_img_sub(yx_shape, df_chunk, mag, mult, diffraction_limit)
                   for df_chunk, mult in zip(df_chunks, mult_chunks)]
    lazy_result = dask.array.stack(
        [dask.array.from_delayed(l, np.array(yx_shape) * mag, np.float) for l in lazy_result])
    return lazy_result.sum(0)


def gen_img(yx_shape, df, mag=10, cmap="gist_rainbow", weight=None, diffraction_limit=False, numthreads=1, hist=False, colorcode="z0"):
    """Generate a 2D image, optionally with z color coding

    Parameters
    ----------
    yx_shape : tuple
        The shape overwhich to render the scene
    df : DataFrame
        A DataFrame object containing localization data
    mag : int
        The magnification factor to render the scene
    cmap : "hsv"
        The color coding for the z image, if set to None, only 2D
        will be rendered
    weight : str
        The key with which to weight the z image, valid options are
        "amp" and "nphotons"
    numthreads : int
        The number of threads to use during rendering. (Experimental)
    hist : bool
        Whether to use gaussian rendering or histogram
    colorcode : string
        key that exists in the data frame such that we can color code by any value.

    Returns
    -------
    img : ndarray
        If no cmap is specified then the result is a 2D array
        If a cmap is specified then the result is a 3D array where
        the last axis is RGBA. the A channel is just the intensity
        It will not have gamma or clipping applied.
    """
    if hist:
        # generate the bins for the histogram
        bins = [np.arange(0, dim + 1.0 / mag, 1 / mag) - 1 / mag / 2 for dim in yx_shape]
        # there's no weighting by amplitude or anything here
        w = np.ones(len(df))

        def func(weights):
            """This is the histogram function"""
            
            @dask.delayed
            def lazy_hist(sample, bins=10, range=None, normed=False, weights=None):
                return np.histogramdd(sample, bins, range, normed, weights)[0]
            
            l = lazy_hist(df[["y0", "x0"]].values, bins, weights=weights)
            return dask.array.from_delayed(l, np.array(yx_shape) * mag, np.float)
    else:
        # here want to weight by sigma_z just like we do with sigmas when generating the gaussians
        w = (1 / (np.sqrt(2 * np.pi)) / df["sigma_z"]).values

        def func(weights):
            """This is the gaussian renderer"""
            return _gen_img_sub_threaded(yx_shape, df, mag, weights, diffraction_limit, numthreads)

    # Generate the intensity image
    img_w = func(w)
    if cmap is not None:
        # calculate the weighting for each localization
        if weight is not None:
            w *= df[weight].values
        # normalize z into the range of 0 to 1
        norm_z = scale(df[colorcode].values)
        # Calculate weighted colors for each z position
        wz = (w[:, None] * matplotlib.cm.get_cmap(cmap)(norm_z))
        # generate the weighted r, g, anb b images
        img_wz_r = func(wz[:, 0])
        img_wz_g = func(wz[:, 1])
        img_wz_b = func(wz[:, 2])
        # combine the images and divide by weights to get a depth-coded RGB image
        rgb = dask.array.dstack((img_wz_r, img_wz_g, img_wz_b)) / img_w[..., None]
        # where weight is 0, replace with 0
        rgb[~np.isfinite(rgb)] = 0
        # add on the alpha img
        rgba = dask.array.dstack((rgb, img_w))
        return DepthCodedImage(rgba.compute(), cmap, mag, (df[colorcode].min(), df[colorcode].max()))
    else:
        # just return the alpha.
        return img_w.compute()


def depthcodeimage(data, cmap="gist_rainbow", projection="max"):
    """Generate a 2D image, optionally with z color coding

    Parameters
    ----------
    data : ndarray
        The data to depth code, will depth code along the first axis
    cmap : matplotlib cmap
        The cmap to use during depth coding

    Returns
    -------
    img : ndarray
        If no cmap is specified then the result is a 2D array
        If a cmap is specified then the result is a 3D array where
        the last axis is RGBA. the A channel is just the intensity
        It will not have gamma or clipping applied.
    """
    # Well thread this along the direction perpendicular to z so that we don't eat memory
    # and to speed things up for large volumes
    nz, ny, nx = data.shape
    # we assume in this case that the data _is_ the weights
    # normalize z into the range of 0 to 1
    norm_z = np.linspace(0, 1, nz)
    # Calculate weighted colors for each z position, drop alpha
    wz = matplotlib.cm.get_cmap(cmap)(norm_z)[:, :3]
    # generate the weighted r, g, anb b images
    projection = projection.lower()
    op = getattr(np.ndarray, projection)
    d_max = data.max()
    d_min = data.min()
    @dask.delayed
    def func(d):
        """Mean func of a plane"""
        # convert to float
        d = (d - d_min) / (d_max - d_min)
        alpha = op(d, axis=0)[:, None]
        weighted_d = d[:, None] * wz[..., None]
        # doesn't make a difference for sum (mean) but does for max or min
        rgb = np.rollaxis(op(weighted_d, axis=0), 1)
        rgba = np.hstack((rgb, alpha))
        return rgba
    
    rgba = dask.array.stack([dask.array.from_delayed(func(d), (nx, 4), float) for d in np.rollaxis(data, 1)])
    return DepthCodedImage(rgba.compute(), cmap, 1, (0, 1))


def contrast_enhancement(data, vmax=None, vmin=None, **kwargs):
    """Enhance the color in an RGB image

    https://math.stackexchange.com/questions/906240/algorithms-to-increase-or-decrease-the-contrast-of-an-image"""
    if vmax is None:
        vmax = data.max()
    if vmin is None:
        vmin = data.min()

    a = 1 / (vmax - vmin)
    b = - vmin * a
    return a * data + b


class DepthCodedImage(np.ndarray):
    """A specialty class to handle depth coded images, especially saving and displaying"""

    def __new__(cls, data, cmap, mag, zrange):
        # from https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(data).view(cls)
        # add the new attribute to the created instance
        obj.cmap = cmap
        obj.mag = mag
        obj.zrange = zrange
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.cmap = getattr(obj, 'cmap', None)
        self.mag = getattr(obj, 'mag', None)
        self.zrange = getattr(obj, 'zrange', None)

    def save(self, savepath):
        """Save data and metadata to a tif file"""
        info_dict = dict(
            cmap=self.cmap,
            mag=self.mag,
            zrange=self.zrange
        )

        tif_kwargs = dict(
            imagej=True, resolution=(self.mag, self.mag),
            metadata=dict(
                # let's stay agnostic to units for now
                unit="pixel",
                # dump info_dict into string
                info=json.dumps(info_dict),
                axes='YXC'
            )
        )

        tif.imsave(fix_ext(savepath, ".tif"), tif_convert(self), **tif_kwargs)

    @classmethod
    def load(cls, path):
        """Load previously saved data"""
        with tif.TiffFile(path) as file:
            info_dict = json.loads(file.pages[0].imagej_tags["info"])
            data = file.asarray()
        return cls(data, **info_dict)

    @property
    def RGB(self):
        """Return the rgb channels of the image"""
        return np.asarray(self)[..., :3]

    @property
    def alpha(self):
        """Return the alpha channel of the image"""
        return np.asarray(self)[..., 3]

    def _norm_alpha(self, auto=False, **kwargs):
        """"""
        # power norm will normalize alpha to 0 to 1 after applying
        # a gamma correction and limiting data to vmin and vmax
        pkwargs = dict(gamma=1, clip=True)
        pkwargs.update(kwargs)
        new_alpha = PowerNorm(**pkwargs)(self.alpha)
        # if auto is requested perform it
        if auto:
            vdict = auto_adjust(new_alpha)
        else:
            vdict = dict()
        
        new_alpha = Normalize(clip=True, **vdict)(new_alpha)
        return new_alpha

    def _norm_data(self, alpha, contrast=False, **kwargs):
        if contrast:
            # do contrast enhancement only, if you want contrast
            # and gamma you'll have to do it iteratively.
            # d2 = d._norm_data(alpha=False, contrast=False, **kwargs)
            # d3 = contrast_enhancement(d2, **kwargs2)
            new_data = contrast_enhancement(self.RGB, **kwargs)
        else:
            new_alpha = self._norm_alpha(**kwargs)
            if alpha:
                new_data = np.dstack((self.RGB, new_alpha))
            else:
                new_data = self.RGB * new_alpha[..., None]
        return new_data

    def save_color(self, savepath, alpha=False, **kwargs):
        """Save a color image of the depth coded data

        Note: to save RGB only use "gamma=0"""
        # normalize path name to make sure that it end's in .tif
        ext = os.path.splitext(savepath)[1]
        if ext.lower() == ".tif":
            alpha = False
        norm_data = self._norm_data(alpha, **kwargs)
        img8bit = (norm_data * 255).astype(np.uint8)
        if ext.lower() == ".tif":
            DepthCodedImage(img8bit, self.cmap, self.mag, self.zrange).save(savepath)
        else:
            imsave(savepath, img8bit)

    def save_alpha(self, savepath, **kwargs):
        # normalize path name to make sure that it end's in .tif
        DepthCodedImage(self.alpha, self.cmap, self.mag, self.zrange).save(savepath)
        
    def plot(self, pixel_size=0.13, unit="μm", scalebar_size=None, subplots_kwargs=dict(), norm_kwargs=dict()):
        """Make a nice plot of the data, with a scalebar"""
        # make the figure and axes
        fig, ax = plt.subplots(**subplots_kwargs)
        # make the colorbar plot
        zdata = np.linspace(self.zrange[0], self.zrange[1], 256).reshape(16, 16)
        cbar = ax.matshow(zdata, zorder=-2, cmap=self.cmap)
        ax.matshow(np.zeros_like(zdata), zorder=-1, cmap="Greys_r")
        # show the color data
        ax.imshow(self._norm_data(False, **norm_kwargs), interpolation=None, zorder=0)
        ax.set_facecolor("k")
        # add the colorbar
        fig.colorbar(cbar, label="z ({})".format(unit), ax=ax, pad=0.01)
        # add scalebar if requested
        if scalebar_size:
            # make sure the length makes sense in data units
            scalebar_length = scalebar_size * self.mag / pixel_size
            default_scale_bar_kwargs = dict(
                loc='lower left',
                pad=0.5,
                color='white',
                frameon=False,
                size_vertical=scalebar_length / 10,
                fontproperties=fm.FontProperties(size="large", weight="bold")
            )
            scalebar = AnchoredSizeBar(ax.transData,
                                       scalebar_length,
                                       '{} {}'.format(scalebar_size, unit),
                                       **default_scale_bar_kwargs
                                       )
            # add the scalebar
            ax.add_artist(scalebar)
        # remove ticks
        ax.set_yticks([])
        ax.set_xticks([])
        # return fig and ax for further processing
        return fig, ax


@dask.delayed
def _gen_zplane(yx_shape, df, zplane, mag, diffraction_limit):
    """A subfunction to generate a single z plane"""
    # again a hard coded radius
    radius = 5
    # find the fiducials worth rendering
    df_zplane = df[np.abs(df.z0 - zplane) < df.sigma_z * radius]
    # calculate the amplitude of the z gaussian.
    amps = np.exp(-((df_zplane.z0 - zplane) / df_zplane.sigma_z) ** 2 / 2) / (np.sqrt(2 * np.pi) * df_zplane.sigma_z)
    # generate a 2D image weighted by the z gaussian.
    toreturn = _jit_gen_img_sub(yx_shape, df_zplane[["y0", "x0", "sigma_y", "sigma_x"]].values, mag, amps.values, diffraction_limit)
    # remove all temporaries
    gc.collect()
    return toreturn


def gen_img_3d(yx_shape, df, zplanes, mag, diffraction_limit, num_workers=None):
    """Generate a 3D image with gaussian point clouds

    Parameters
    ----------
    yx_shape : tuple
        The shape overwhich to render the scene
    df : DataFrame
        A DataFrame object containing localization data
    zplanes : array
        The planes at which the user wishes to render
    mag : int
        The magnification factor to render the scene"""
    new_shape = tuple(np.array(yx_shape) * mag)
    # print(dask.array.from_delayed(_gen_zplane(df, yx_shape, zplanes[0], mag), new_shape, np.float))
    rendered_planes = [dask.array.from_delayed(_gen_zplane(yx_shape, df, zplane, mag, diffraction_limit), new_shape, np.float)
                                   for zplane in zplanes]
    to_compute = dask.array.stack(rendered_planes)
    return to_compute.compute(num_workers=num_workers)


def save_img_3d(yx_shape, df, savepath, zspacing=None, zplanes=None, mag=10, diffraction_limit=False,
                hist=False, num_workers=None, **kwargs):
    """Generates and saves a gaussian rendered 3D image along with the relevant metadata in a tif stack

    Parameters
    ----------
    yx_shape : tuple
        The shape overwhich to render the scene
    df : DataFrame
        A DataFrame object containing localization data
    savepath : str
        the path to save the file in.
    mag : int
        The magnification factor to render the scene


    https://stackoverflow.com/questions/10724495/getting-all-arguments-and-values-passed-to-a-python-function
    """
    # figure out the zplanes to calculate
    if zplanes is None:
        if zspacing is None:
            raise ValueError("zspacing or zplanes must be specified")
        # this is better I think.
        # zplanes = np.arange(df.z0.min() + zspacing, df.z0.max() + zspacing, zspacing) - zspacing / 2
        zplanes = np.arange(df.z0.min(), df.z0.max() + zspacing, zspacing)

    # generate the actual image
    if not hist:
        img3d = gen_img_3d(yx_shape, df, zplanes, mag, diffraction_limit, num_workers=num_workers)
    else:
        dz = np.diff(zplanes)
        bins = [np.concatenate((zplanes[:-1] - dz / 2, zplanes[-2:] + dz[-1] / 2))]
        # we want the xy bins to be centered on each pixel, i.e. the first bin is centered
        # on 0 so has edges of -0.5 and 0.5
        bins = bins + [np.arange(0, dim + 1.0 / mag, 1 / mag) - 1 / mag / 2 for dim in yx_shape]
        img3d = fast_hist3d(df[["z0", "y0", "x0"]].values, bins)[0]
    # save kwargs
    tif_kwargs = dict(resolution=(mag, mag),
        metadata=dict(
            # spacing is the depth spacing for imagej
            spacing=zspacing,
            # let's stay agnostic to units for now
            unit="pixel",
            # we want imagej to interpret the image as a z-stack
            # so set slices to the length of the image
            slices=len(img3d),
            # This information is mostly redundant with "spacing" but is included
            # incase one wanted to render arbitrarily spaced planes.
            labels=["z = {}".format(zplane) for zplane in zplanes],
            axes="ZYX"
            )
        )

    tif_ready = tif_convert(img3d)
    # check if bigtiff is necessary.
    if tif_ready.nbytes / (4 * 1024**3) < 0.95:
        tif_kwargs.update(dict(imagej=True))
    else:
        tif_kwargs.update(dict(imagej=False, compress=6, bigtiff=True))

    # incase user wants to change anything
    tif_kwargs.update(kwargs)
    # save the tif
    tif.imsave(fix_ext(savepath, ".tif"), tif_ready, **tif_kwargs)

    # return data to user for further processing.
    return img3d


def fix_ext(path, ext):
    if os.path.splitext(path)[1].lower() != ext.lower():
        path += ext
    return path
