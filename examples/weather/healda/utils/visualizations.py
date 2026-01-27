# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import types

import cartopy.crs
import cartopy.feature
import matplotlib.pyplot as plt
import numpy as np
import torch
from earth2grid import healpix
from matplotlib.colors import LogNorm

# Parameters from the proj4 string
central_longitude = -85.0
central_latitude = 30.0
standard_parallels = (23.0, 37.0)

# Create the LambertConformalConic projection
caribbean_se_us_proj = cartopy.crs.LambertConformal(
    central_longitude=central_longitude,
    central_latitude=central_latitude,
    standard_parallels=standard_parallels,
    globe=cartopy.crs.Globe(ellipse="WGS84"),
)


_projections = {
    "PlateCarree": cartopy.crs.PlateCarree(),
    "Robinson": cartopy.crs.Robinson(),
    "Robinson_180": cartopy.crs.Robinson(180),
    "conus": cartopy.crs.epsg(5069),
    "south_pole": cartopy.crs.SouthPolarStereo(),
    "north_pole": cartopy.crs.NorthPolarStereo(),
    "carib": caribbean_se_us_proj,
    "warm_pool": cartopy.crs.PlateCarree(),
    "conus_large": cartopy.crs.PlateCarree(),
}

_extents = {
    "carib": [-100, -60, 10, 40],
    "warm_pool": [85, 160, -15, 20],
    "conus_large": [-130, -60, 18, 50],
}


def get_lim(target_crs, extents):
    """Return axis limits for the given CRS and optional lat/lon extents."""
    if target_crs in [cartopy.crs.SouthPolarStereo(), cartopy.crs.NorthPolarStereo()]:
        lim = 4_500_000
        return -lim, lim, -lim, lim
    elif extents:
        return latlon_to_grid_extents(target_crs, extents)
    else:
        return target_crs.x_limits + target_crs.y_limits


def create_regular_grid_in_projection(projection, nx, ny, extents=None):
    """
    Create a regular grid of lat-lon coordinates in a given Cartopy projection.

    Parameters:
    projection (cartopy.crs.Projection): The desired Cartopy projection
    resolution (float): The grid resolution in projection units

    Returns:
    tuple: Two 2D arrays, one for latitudes and one for longitudes
    """
    # Get the projection's limits
    x_min, x_max, y_min, y_max = get_lim(projection, extents)
    # Create a regular grid in the projection coordinates
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(x, y)

    # Transform the gridded coordinates back to lat-lon
    geodetic = cartopy.crs.Geodetic()
    transformed = geodetic.transform_points(projection, xx, yy)

    lons = transformed[..., 0]
    lats = transformed[..., 1]

    # Filter out invalid points (those outside the projection's valid domain)
    valid = np.logical_and(np.isfinite(lons), np.isfinite(lats))
    lons[~valid] = np.nan
    lats[~valid] = np.nan

    return lats, lons, xx, yy


def latlon_to_grid_extents(proj, extents):
    lon_min, lon_max, lat_min, lat_max = extents
    # Create arrays of lat-lon points
    lons = np.array([lon_min, lon_max, lon_max, lon_min])
    lats = np.array([lat_min, lat_min, lat_max, lat_max])

    # Transform to projection coordinates
    x, y = proj.transform_points(cartopy.crs.PlateCarree(), lons, lats).T[:2]

    # Get the min and max x and y values
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    return x_min, x_max, y_min, y_max


def visualize(
    x,
    region="Robinson",
    projection=None,
    extents=None,
    nest=False,
    hpxpad=False,
    pos=None,
    n=None,
    title=None,
    colorbar_label=None,
    cmap=None,
    coastlines_color="k",
    add_colorbar=True,
    extend=None,
    nlat=256,
    nlon=512,
    border=False,
    kind="pcolormesh",
    interp_method="bilinear",
    **kw,
):
    if x.ndim != 1:
        raise ValueError(f"Expected 1D input but received {x.ndim}.")

    if projection is None:
        crs = _projections[region]
        extents = _extents.get(region, None)
    else:
        crs = projection

    if nest:
        pixel_order = healpix.NEST
    elif hpxpad:
        pixel_order = healpix.HEALPIX_PAD_XY
    else:
        pixel_order = healpix.RING

    hpx = healpix.Grid(healpix.npix2level(x.shape[-1]), pixel_order=pixel_order)
    lat, lon, xx, yy = create_regular_grid_in_projection(
        crs, nlat, nlon, extents=extents
    )
    cmap = plt.get_cmap(cmap, n)
    mask = ~np.isnan(lat)
    latm = lat[mask]
    lonm = lon[mask]
    x = torch.as_tensor(x)
    out = torch.zeros_like(torch.tensor(lat)).to(x)
    if interp_method == "bilinear":
        regrid = hpx.get_bilinear_regridder_to(latm, lonm)
        regrid.to(x)
        out[mask] = regrid(x)
        out[~mask] = torch.nan
    elif interp_method == "bin":
        pix = hpx.ang2pix(
            torch.as_tensor(lonm, device=x.device),
            torch.as_tensor(latm, device=x.device),
        )
        out[mask] = x[pix]
        out[~mask] = torch.nan
    else:
        raise ValueError(f"Invalid interpolation method: {interp_method}")

    if isinstance(pos, tuple):
        subplot_args = pos
    elif pos is not None:
        subplot_args = (pos,)
    else:
        subplot_args = ()

    ax = plt.subplot(*subplot_args, projection=crs)
    plot_func = getattr(ax, kind)
    im = plot_func(xx, yy, out.cpu(), transform=crs, cmap=cmap, **kw)
    ax.coastlines(color=coastlines_color)

    if border:
        ax.add_feature(cartopy.feature.BORDERS, linewidth=1, edgecolor="lightgray")
        ax.add_feature(cartopy.feature.STATES, linewidth=0.4, edgecolor="lightgray")

    cb = None
    if add_colorbar:
        cb = plt.colorbar(im, orientation="horizontal", extend=extend)
        if colorbar_label:
            cb.set_label(colorbar_label)
    if title:
        ax.set_title(title)

    return types.SimpleNamespace(ax=ax, im=im, cb=cb)


def plot_in_lat_lon(
    ax,
    x,
    lat,
    lon,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    pr,
    regrid,
    vmin,
    vmax,
    nest=False,
    hpxpad=False,
):
    if regrid:
        if x.ndim != 1:
            raise ValueError(f"Expected 1D input but received {x.ndim}.")
        if nest:
            pixel_order = healpix.NEST
        if hpxpad:
            pixel_order = healpix.HEALPIX_PAD_XY
        else:
            pixel_order = healpix.RING
        hpx = healpix.Grid(healpix.npix2level(x.shape[-1]), pixel_order=pixel_order)
        mask = ~np.isnan(lat)
        latm = lat[mask]
        lonm = lon[mask]
        x = torch.as_tensor(x)
        regrid = hpx.get_bilinear_regridder_to(latm, lonm)
        regrid.to(x)
        data = torch.zeros_like(torch.tensor(lat)).to(x)
        data[mask] = regrid(x)
        data[~mask] = torch.nan
    else:
        data = x
    lon_max = max(lon_max, np.min(np.max(lon, axis=1)))
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=cartopy.crs.PlateCarree())
    if pr:
        data[data <= np.exp(-12)] = np.exp(-12)
        vmin = np.exp(-12)
        vmax = np.exp(-4)
        mesh = ax.pcolormesh(
            lon,
            lat,
            data,
            transform=cartopy.crs.PlateCarree(),
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="magma",
        )
    else:
        mesh = ax.pcolormesh(
            lon,
            lat,
            data,
            transform=cartopy.crs.PlateCarree(),
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
    mesh.set_rasterized(True)
    ax.coastlines(resolution="110m", linewidth=1, color="lightgray")
    ax.add_feature(cartopy.feature.BORDERS, linewidth=1, edgecolor="lightgray")
    ax.add_feature(cartopy.feature.STATES, linewidth=0.4, edgecolor="lightgray")
    return mesh
