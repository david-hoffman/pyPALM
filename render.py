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
from dphutils import _calc_pad, scale, get_git
from dphplotting import auto_adjust

import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.colors import Normalize, PowerNorm, ListedColormap
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from skimage.external import tifffile as tif
from imageio import imwrite
from PIL.PngImagePlugin import PngInfo

# get multiprocessing support
import dask
import dask.array

import logging

logger = logging.getLogger()

greys_alpha_cm = ListedColormap([(i / 255,) * 3 + ((255 - i) / 255,) for i in range(256)])

general_meta_data = {"git revision": get_git(os.path.split(__file__)[0]), "module": __name__}

image_software = (
    tif.__name__
    + "-"
    + tif.__version__
    + "+"
    + general_meta_data["module"]
    + "-"
    + general_meta_data["git revision"]
)

general_meta_data["image_software"] = image_software

logger.info(general_meta_data)


def choose_dtype(max_value):
    """choose the appropriate dtype for saving images"""
    # if any type of float, use float32
    if np.issubdtype(max_value, np.inexact):
        return np.float32
    # check for integers now
    if max_value < 2 ** 8:
        return np.uint8
    elif max_value < 2 ** 16:
        return np.uint16
    elif max_value < 2 ** 32:
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
def jit_hist3d_with_weights(zpositions, ypositions, xpositions, weights, shape):
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
                "The dimension of bins must be equal to the dimension of the" " sample x."
            )
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
            raise ValueError("myrange parameter must be finite.")
        smin = np.zeros(D)
        smax = np.zeros(D)
        for i in range(D):
            smin[i], smax[i] = myrange[i]

    # Make sure the bins have a finite width.
    for i in range(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - 0.5
            smax[i] = smax[i] + 0.5

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
                    "Element at index %s in `bins` should be a positive " "integer." % i
                )
            nbin[i] = bins[i] + 2  # +2 for outlier bins
            edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1, dtype=edge_dt)
        else:
            edges[i] = np.asarray(bins[i], edge_dt)
            nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
        dedges[i] = np.diff(edges[i]).min()
        if np.any(np.asarray(dedges[i]) <= 0):
            raise ValueError(
                "Found bin edge of size <= 0. Did you specify `bins` with"
                "non-monotonic sequence?"
            )

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
            not_smaller_than_edge = sample[:, i] >= edges[i][-1]
            on_edge = np.around(sample[:, i], decimal) == np.around(edges[i][-1], decimal)
            # Shift these points one bin to the left.
            Ncount[i][np.where(on_edge & not_smaller_than_edge)[0]] -= 1

    if weights is not None:
        weights = np.asarray(weights)
        hist = jit_hist3d_with_weights(*Ncount, weights=weights, shape=shape)
    else:
        hist = jit_hist3d(*Ncount, shape=shape)
    return hist, edges


### Gaussian Rendering
def _gauss(yw, xw, y0, x0, sy, sx):
    """Simple normalized 2D gaussian function for rendering"""
    # for this model, x and y are seperable, so we can generate
    # two gaussians and take the outer product
    y = np.arange(yw)
    x = np.arange(xw)
    amp = 1 / (2 * np.pi * sy * sx)
    gy = np.exp(-(((y - y0) / sy) ** 2) / 2)
    gx = np.exp(-(((x - x0) / sx) ** 2) / 2)
    return amp * np.outer(gy, gx)


_jit_gauss = njit(_gauss, nogil=True)


def _gen_img_sub(yx_shape, params, mag, multipliers, diffraction_limit):
    # hard coded radius, this is really how many sigmas you want
    # to use to render each gaussian
    radius = 3

    # initialize the image
    ymax = int(yx_shape[0] * mag)
    xmax = int(yx_shape[1] * mag)
    img = np.zeros((ymax, xmax))
    # iterate through all localizations
    for i in range(len(params)):
        if not np.isfinite(params[i]).all():
            # skip nans
            continue
        # pull parameters adjust to new magnification (Numba requires that we expand this out, explicitly)
        y0 = params[i, 0] * mag
        x0 = params[i, 1] * mag
        sy = params[i, 2] * mag
        sx = params[i, 3] * mag
        # adjust parameters if diffraction limit is requested
        if diffraction_limit:
            sy = max(sy, 0.5)
            sx = max(sx, 0.5)
        # calculate the render window size
        wy = int(np.rint(sy * radius * 2.0))
        wx = int(np.rint(sx * radius * 2.0))
        # calculate the area in the image
        ystart = int(np.rint(y0)) - wy // 2
        yend = ystart + wy
        xstart = int(np.rint(x0)) - wx // 2
        xend = xstart + wx
        if yend <= 0 or xend <= 0:
            # make sure we have nothing negative
            continue
        # don't go over the edge
        yend = min(yend, ymax)
        ystart = max(ystart, 0)
        xend = min(xend, xmax)
        xstart = max(xstart, 0)
        wy = yend - ystart
        wx = xend - xstart
        if wy <= 0 or wx <= 0:
            # make sure there is something to render
            continue
        # adjust coordinates to window coordinates
        y1 = y0 - ystart
        x1 = x0 - xstart
        # generate gaussian
        g = _jit_gauss(wy, wx, y1, x1, sy, sx)
        # weight if requested
        if len(multipliers):
            g *= multipliers[i]
        # update image
        img[ystart:yend, xstart:xend] += g
    return img


_jit_gen_img_sub = njit(_gen_img_sub, nogil=True)


def gen_img(
    yx_shape,
    df,
    mag=10,
    cmap="gist_rainbow",
    weight=None,
    diffraction_limit=False,
    numthreads=1,
    hist=False,
    colorcode="z0",
    zscaling=None,
):
    """Generate a 2D image, optionally with z color coding

    Parameters
    ----------
    yx_shape : tuple
        The shape overwhich to render the scene
    df : DataFrame
        A DataFrame object containing localization data
    mag : int
        The magnification factor to render the scene
    cmap : "gist_rainbow"
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

    new_shape = np.array(yx_shape) * mag

    # calculate the weighting for each localization
    if weight is not None:
        try:
            w = df[weight].to_numpy()
        except KeyError:
            w = weight
    else:
        w = np.ones(len(df))

    if hist:
        # generate the bins for the histogram
        bins = [np.arange(0, dim + 1.0 / mag, 1 / mag) - 1 / mag / 2 for dim in yx_shape]
        # there's no weighting by amplitude or anything here

        def func(weights):
            """This is the histogram function"""
            lazy_hist = dask.delayed(np.histogramdd)(
                df[["y0", "x0"]].values, bins=bins, density=False, weights=weights
            )[0]
            return lazy_hist

    else:
        # if requested limit sigma_z values to that there aren't super bright guys.
        # min_sigma_z = 1 / np.sqrt(2 * np.pi)
        # if diffraction_limit and zscaling is not None:
        #     min_sigma_z = 0.5 * zscaling / mag

        # here want to weight by sigma_z just like we do with sigmas when generating the gaussians
        # w = 1 / (np.sqrt(2 * np.pi) * np.fmax(df["sigma_z"].values, min_sigma_z))

        def func(weights):
            """This is the gaussian renderer"""
            keys_for_render = ["y0", "x0", "sigma_y", "sigma_x"]
            df_arr = df[keys_for_render].values

            @dask.delayed
            def delayed_jit_gen_img_sub(chunk):
                if chunk is not None:
                    return _jit_gen_img_sub(
                        yx_shape, df_arr[chunk], mag, weights[chunk], diffraction_limit
                    )
                return _jit_gen_img_sub(yx_shape, df_arr, mag, weights, diffraction_limit)

            if numthreads > 1:
                chunklen = (len(df) + numthreads - 1) // numthreads
                chunks = [slice(i * chunklen, (i + 1) * chunklen) for i in range(numthreads)]
                # Create argument tuples for each input chunk
                lazy_result = [delayed_jit_gen_img_sub(chunk) for chunk in chunks]
            else:
                lazy_result = delayed_jit_gen_img_sub(None)

            return lazy_result

    # Generate the intensity image
    img_w = func(w)
    if cmap is not None:
        # normalize z into the range of 0 to 1
        norm_z = scale(df[colorcode].values)
        # Calculate weighted colors for each z position
        wz = w[:, None] * matplotlib.cm.get_cmap(cmap)(norm_z)
        # generate the weighted r, g, anb b images
        img_wz_r = func(wz[:, 0])
        img_wz_g = func(wz[:, 1])
        img_wz_b = func(wz[:, 2])
        # compute and rearrange
        darr = np.asarray(dask.compute(img_wz_r, img_wz_g, img_wz_b, img_w))
        if darr.ndim > 3:
            darr = darr.sum(1)
        img_wz_r, img_wz_g, img_wz_b, img_w = darr
        # combine the images and divide by weights to get a depth-coded RGB image
        with np.errstate(invalid="ignore"):
            rgb = np.dstack((img_wz_r, img_wz_g, img_wz_b)) / img_w[..., None]
        # where weight is 0, replace with 0
        rgb[~np.isfinite(rgb)] = 0
        # add on the alpha img
        rgba = np.dstack((rgb, img_w))
        return DepthCodedImage(rgba, cmap, mag, (df[colorcode].min(), df[colorcode].max()))
    else:
        # just return the alpha.
        img_w = np.asarray(dask.compute(img_w))
        if img_w.ndim > 2:
            img_w = img_w.sum(0)
        return img_w


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

    rgba = dask.array.stack(
        [dask.array.from_delayed(func(d), (nx, 4), float) for d in np.rollaxis(data, 1)]
    )
    return DepthCodedImage(rgba.compute(), cmap, 1, (0, 1))


def contrast_enhancement(data, vmax=None, vmin=None, **kwargs):
    """Enhance the color in an RGB image

    https://math.stackexchange.com/questions/906240/algorithms-to-increase-or-decrease-the-contrast-of-an-image"""
    if vmax is None:
        vmax = data.max()
    if vmin is None:
        vmin = data.min()

    a = 1 / (vmax - vmin)
    b = -vmin * a
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
        self.cmap = getattr(obj, "cmap", None)
        self.mag = getattr(obj, "mag", None)
        self._zrange = getattr(obj, "_zrange", None)

    @property
    def zrange(self):
        """The z range of the depth coded image"""
        return self._zrange

    @zrange.setter
    def zrange(self, new):
        """Make sure the zrange is not a numpy object"""
        zmin, zmax = new
        # convert to float so that JSON can serialize them
        self._zrange = float(zmin), float(zmax)

    @property
    def info_dict(self):
        info_dict = dict(cmap=self.cmap, mag=self.mag, zrange=self.zrange)
        return info_dict

    def save(self, savepath):
        """Save data and metadata to a tif file"""

        tif_kwargs = dict(
            imagej=True,
            resolution=(self.mag, self.mag),
            metadata=dict(
                # let's stay agnostic to units for now
                unit="pixel",
                # dump info_dict into string
                info=json.dumps(self.info_dict),
                axes="YXC",
            ),
            software=image_software,
        )

        tif_kwargs["metadata"].update(general_meta_data)
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
        """Adjust alpha with Gamma and min max"""
        # power norm will normalize alpha to 0 to 1 after applying
        # a gamma correction and limiting data to vmin and vmax
        # if auto is requested perform it
        if auto:
            myalpha = self.alpha[self.alpha > 0]
            vmin, vmax = np.percentile(myalpha, (1, 99))
            vdict = dict(vmin=vmin, vmax=vmax)
        else:
            vdict = dict()
        pkwargs = dict(gamma=1, clip=True)
        pkwargs.update(vdict)
        pkwargs.update(kwargs)
        new_alpha = PowerNorm(**pkwargs)(self.alpha)

        new_alpha = Normalize(clip=True)(new_alpha)
        return new_alpha

    def _norm_data(self, alpha, contrast=False, **kwargs):
        """Norm the data

        Parameters:
        alpha : bool
            Include alpha channel or not
        contrast : bool
            maximize contrast using `contrast_enhancement`

        kwargs are passed to DepthCodedImage._norm_alpha
        """
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
        img8bit = (np.clip(norm_data, 0, 1) * 255).astype(np.uint8)
        if ext.lower() == ".tif":
            DepthCodedImage(img8bit, self.cmap, self.mag, self.zrange).save(savepath)
        else:
            imwrite_kwargs = {}
            if os.path.splitext(savepath)[-1].lower() == ".png":
                pnginfo = PngInfo()

                def add_text(k, v):
                    """Stringify all the things"""
                    pnginfo.add_text(str(k), str(v))

                for k, v in self.info_dict.items():
                    add_text(k, v)

                add_text("software", image_software)
                imwrite_kwargs["pnginfo"] = pnginfo

            imwrite(savepath, img8bit, **imwrite_kwargs)

    def save_alpha(self, savepath, **kwargs):
        # normalize path name to make sure that it end's in .tif
        DepthCodedImage(self.alpha, self.cmap, self.mag, self.zrange).save(savepath)

    def plot(
        self,
        pixel_size=0.13,
        unit="μm",
        scalebar_size=None,
        subplots_kwargs=dict(),
        norm_kwargs=dict(),
    ):
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
                loc="lower left",
                pad=0.5,
                color="white",
                frameon=False,
                size_vertical=scalebar_length / 10,
                fontproperties=fm.FontProperties(size="large", weight="bold"),
            )
            scalebar = AnchoredSizeBar(
                ax.transData,
                scalebar_length,
                "{} {}".format(scalebar_size, unit),
                **default_scale_bar_kwargs
            )
            # add the scalebar
            ax.add_artist(scalebar)
        # remove ticks
        ax.set_yticks([])
        ax.set_xticks([])
        # return fig and ax for further processing
        return fig, ax


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
    if diffraction_limit:
        # we need to make sure that we make sure our sigma_z is large enoughso we don't have the
        # equivalent of puncta, or worse, miss localizations entirely
        # take the minimum spacing
        try:
            zspacing = np.diff(zplanes).min()
        except ValueError:
            # there was only one value
            zspacing = 0
        # copy our data so we don't affect original
        df = df[["z0", "y0", "x0", "sigma_z", "sigma_y", "sigma_x"]]
        # set min sigma_z to half min zspacing
        df.sigma_z = np.fmax(zspacing * 0.5, df.sigma_z)

    @dask.delayed
    def _gen_zplane(zplane):
        """A subfunction to generate a single z plane"""
        # again a hard coded radius
        radius = 3
        # find the fiducials worth rendering
        df_zplane = df[np.abs(df.z0 - zplane) < df.sigma_z * radius]
        # calculate the amplitude of the z gaussian.
        amps = np.exp(-(((df_zplane.z0 - zplane) / df_zplane.sigma_z) ** 2) / 2) / (
            np.sqrt(2 * np.pi) * df_zplane.sigma_z
        )
        # generate a 2D image weighted by the z gaussian.
        toreturn = _jit_gen_img_sub(
            yx_shape,
            df_zplane[["y0", "x0", "sigma_y", "sigma_x"]].values,
            mag,
            amps.values,
            diffraction_limit,
        )
        # remove all temporaries
        del df_zplane
        del amps
        gc.collect()
        return toreturn

    # Build delayed array
    rendered_planes = [
        dask.array.from_delayed(_gen_zplane(zplane), new_shape, np.float) for zplane in zplanes
    ]
    to_compute = dask.array.stack(rendered_planes)
    return to_compute.compute(num_workers=num_workers)


def save_img_3d(
    yx_shape,
    df,
    savepath,
    zspacing=None,
    zplanes=None,
    mag=10,
    diffraction_limit=False,
    hist=False,
    num_workers=None,
    **kwargs
):
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
        # we're doing a gaussian rendering
        img3d = gen_img_3d(yx_shape, df, zplanes, mag, diffraction_limit, num_workers=num_workers)
    else:
        dz = np.diff(zplanes)
        bins = [np.concatenate((zplanes[:-1] - dz / 2, zplanes[-2:] + dz[-1] / 2))]
        # we want the xy bins to be centered on each pixel, i.e. the first bin is centered
        # on 0 so has edges of -0.5 and 0.5
        bins = bins + [np.arange(0, dim + 1.0 / mag, 1 / mag) - 1 / mag / 2 for dim in yx_shape]
        img3d = fast_hist3d(df[["z0", "y0", "x0"]].values, bins)[0]

    # if user provides save path save image as well as return to user
    if savepath:
        # save kwargs
        tif_kwargs = dict(
            resolution=(mag, mag),
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
                axes="ZYX",
            ),
            software=image_software,
        )

        tif_kwargs["metadata"].update(general_meta_data)

        tif_ready = tif_convert(img3d)
        # check if bigtiff is necessary.
        if tif_ready.nbytes / (4 * 1024 ** 3) < 0.95:
            tif_kwargs.update(dict(imagej=True))
        else:
            tif_kwargs.update(dict(imagej=False, bigtiff=True))

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
