from __future__ import annotations
from typing import Any
import time
import zarr
import zarr.storage
from zarr.storage._fsspec import _make_async
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
    consolidated: bool = False,
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
    consolidated:
        Write consolidated metadata after all levels are written.
        Ignored for zarr v3 (consolidated metadata is not part of the spec).
    """
    root_path = path.rstrip("/")
    async_fs = _make_async(fs)

    def _level_store(level: int) -> zarr.storage.FsspecStore:
        return zarr.storage.FsspecStore(async_fs, path=f"{root_path}/{level}")

    def _write_level(ds: xr.Dataset, level: int, enc: dict) -> None:
        ds.to_zarr(
            _level_store(level),
            mode="a",
            encoding=enc,
            zarr_format=zarr_format,
            consolidated=False,
        )
        zarr.consolidate_metadata(_level_store(level))

    def _open_level(level: int) -> xr.Dataset:
        return xr.open_zarr(_level_store(level), consolidated=True)

    n_levels = len(pyramid.encoding)

    root = zarr.open_group(
        zarr.storage.FsspecStore(async_fs, path=root_path),
        mode=mode,
        zarr_format=zarr_format,
    )
    root.attrs.update(pyramid.dt.attrs)

    ds0 = pyramid.dt["/0"].ds
    ny, nx = ds0.sizes.get(y_dim, "?"), ds0.sizes.get(x_dim, "?")
    print(f"[1/{n_levels}] Writing level 0  ({ny} x {nx})", flush=True)
    t0 = time.perf_counter()
    _write_level(ds0, 0, pyramid.encoding["/0"])
    print(f"[1/{n_levels}] Level 0 done  ({time.perf_counter() - t0:.1f}s)", flush=True)

    for i in range(1, n_levels):
        template_ds = pyramid.dt[f"/{i}"].ds
        prev_ds = _open_level(i - 1)

        curr_ds = getattr(
            prev_ds.coarsen({x_dim: 2, y_dim: 2}, boundary="trim"), method
        )()
        curr_ds.attrs = template_ds.attrs
        for var in curr_ds.data_vars:
            if var in template_ds:
                curr_ds[var].attrs = template_ds[var].attrs

        ny, nx = template_ds.sizes.get(y_dim, "?"), template_ds.sizes.get(x_dim, "?")
        print(f"[{i + 1}/{n_levels}] Writing level {i}  ({ny} x {nx})", flush=True)
        t0 = time.perf_counter()
        _write_level(curr_ds, i, pyramid.encoding[f"/{i}"])
        print(
            f"[{i + 1}/{n_levels}] Level {i} done  ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )

    if consolidated and zarr_format != 3:
        zarr.consolidate_metadata(zarr.storage.FsspecStore(async_fs, path=root_path))
