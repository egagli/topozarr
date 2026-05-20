from __future__ import annotations
from typing import Any
import time
import zarr
import xarray as xr


def write_pyramid(
    pyramid: Any,
    fs: Any,
    path: str,
    x_dim: str = "x",
    y_dim: str = "y",
    method: str = "mean",
    mode: str = "w",
    zarr_format: int = 3,
) -> None:
    """Write a pyramid to a zarr store one level at a time, breaking the dask chain.

    ``pyramid.dt.to_zarr()`` submits all pyramid levels as a single dask graph.
    Because each coarser level's graph chains back through every finer level,
    dask may hold many intermediate full-resolution arrays in memory simultaneously,
    causing OOM on large datasets.

    This function avoids that by:
    1. Writing level 0 (finest) directly from the pyramid's DataTree.
    2. For each subsequent level, reading the just-written zarr group back as fresh
       dask arrays (no inherited graph) and coarsening from there.

    After each level is written, consolidated metadata is written so the next
    open bypasses zarr's directory listing (which fails on flat-namespace stores
    such as Azure Blob Storage).

    Parameters
    ----------
    pyramid:
        Pyramid created by ``create_pyramid``.
    fs:
        fsspec filesystem object (e.g. adlfs.AzureBlobFileSystem, gcsfs.GCSFileSystem).
    path:
        Root path within the filesystem for the pyramid (e.g. "container/path/to/pyramid").
    x_dim:
        Name of the x spatial dimension — must match what was passed to ``create_pyramid``.
    y_dim:
        Name of the y spatial dimension — must match what was passed to ``create_pyramid``.
    method:
        Coarsening method — must match what was passed to ``create_pyramid``.
    mode:
        ``'w'`` to overwrite, ``'a'`` to append/update.
    zarr_format:
        Zarr format version (2 or 3).
    """
    root_path = path.rstrip("/")
    n_levels = len(pyramid.encoding)

    root_store = fs.get_mapper(root_path)
    root = zarr.open_group(root_store, mode=mode, zarr_format=zarr_format)
    root.attrs.update(pyramid.dt.attrs)

    ds0 = pyramid.dt["/0"].ds
    ny, nx = ds0.sizes.get(y_dim, "?"), ds0.sizes.get(x_dim, "?")
    print(f"[1/{n_levels}] Writing level 0  ({ny} x {nx})", flush=True)
    t0 = time.perf_counter()
    store0 = fs.get_mapper(f"{root_path}/0")
    ds0.to_zarr(store=store0, mode="a", encoding=pyramid.encoding["/0"],
                zarr_format=zarr_format, consolidated=False)
    zarr.consolidate_metadata(store0)
    print(f"[1/{n_levels}] Level 0 done  ({time.perf_counter() - t0:.1f}s)", flush=True)

    for i in range(1, n_levels):
        template_ds = pyramid.dt[f"/{i}"].ds
        prev_store = fs.get_mapper(f"{root_path}/{i - 1}")
        current_store = fs.get_mapper(f"{root_path}/{i}")

        prev_ds = xr.open_zarr(prev_store, consolidated=True, mask_and_scale=True)
        curr_ds = getattr(prev_ds.coarsen({x_dim: 2, y_dim: 2}, boundary="trim"), method)()
        curr_ds.attrs = template_ds.attrs
        for var in curr_ds.data_vars:
            if var in template_ds:
                curr_ds[var].attrs = template_ds[var].attrs

        ny, nx = template_ds.sizes.get(y_dim, "?"), template_ds.sizes.get(x_dim, "?")
        print(f"[{i + 1}/{n_levels}] Writing level {i}  ({ny} x {nx})", flush=True)
        t0 = time.perf_counter()
        curr_ds.to_zarr(store=current_store, mode="a", encoding=pyramid.encoding[f"/{i}"],
                        zarr_format=zarr_format, consolidated=False)
        zarr.consolidate_metadata(current_store)
        print(
            f"[{i + 1}/{n_levels}] Level {i} done  ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
