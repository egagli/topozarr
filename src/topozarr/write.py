from __future__ import annotations
from typing import Any
import time
import zarr
import xarray as xr


def _is_fsmap(store: Any) -> bool:
    try:
        from fsspec.mapping import FSMap
        return isinstance(store, FSMap)
    except ImportError:
        return False


def _sub_mapper(store: Any, level: int) -> Any:
    return store.fs.get_mapper(store.root.rstrip("/") + "/" + str(level))


def _write_level(
    ds: xr.Dataset,
    store: Any,
    level: int,
    encoding: dict,
    zarr_format: int,
) -> None:
    if _is_fsmap(store):
        ds.to_zarr(
            _sub_mapper(store, level),
            mode="a",
            encoding=encoding,
            zarr_format=zarr_format,
            consolidated=False,
        )
    else:
        ds.to_zarr(
            store,
            group=str(level),
            mode="a",
            encoding=encoding,
            zarr_format=zarr_format,
            consolidated=False,
        )


def _open_level(
    store: Any,
    level: int,
    encoding: dict | None = None,
    template: xr.Dataset | None = None,
) -> xr.Dataset:
    if _is_fsmap(store):
        import numpy as np
        import dask.array as da

        sub = _sub_mapper(store, level)
        zarr_grp = zarr.open_group(sub, mode="r")

        data_vars = {}
        for var in template.data_vars:
            raw = da.from_zarr(zarr_grp[var]).astype("float32")
            fill = (encoding or {}).get(var, {}).get("_FillValue")
            if fill is not None:
                raw = da.where(raw == fill, np.nan, raw)
            data_vars[var] = xr.DataArray(raw, dims=template[var].dims)

        coords = {
            k: (v.compute() if hasattr(v, "compute") else v)
            for k, v in template.coords.items()
        }
        return xr.Dataset(data_vars, coords=coords, attrs=template.attrs)
    else:
        return xr.open_zarr(store, group=str(level), consolidated=False)


def _print_level_start(
    i: int, n_levels: int, ds: xr.Dataset, x_dim: str, y_dim: str
) -> None:
    ny = ds.sizes.get(y_dim, "?")
    nx = ds.sizes.get(x_dim, "?")
    print(f"[{i + 1}/{n_levels}] Writing level {i}  ({ny} x {nx})", flush=True)


def _print_level_done(i: int, n_levels: int, elapsed: float) -> None:
    print(f"[{i + 1}/{n_levels}] Level {i} done  ({elapsed:.1f}s)", flush=True)


def write_pyramid(
    pyramid: Any,
    store: Any,
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
    store:
        Destination zarr store, FSMap, or path string.
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
    n_levels = len(pyramid.encoding)

    # Write root group and its multiscale/spatial attributes into zarr.json.
    # zarr v3 accepts FSMap here because no subgroup path is involved.
    root = zarr.open_group(store, mode=mode, zarr_format=zarr_format)
    root.attrs.update(pyramid.dt.attrs)

    # Level 0 is the finest resolution — write directly from the pyramid DataTree.
    ds0 = pyramid.dt["/0"].ds
    _print_level_start(0, n_levels, ds0, x_dim, y_dim)
    t0 = time.perf_counter()
    _write_level(ds0, store, 0, pyramid.encoding["/0"], zarr_format)
    _print_level_done(0, n_levels, time.perf_counter() - t0)

    # For each subsequent level, read the previous level back from zarr to get
    # fresh dask arrays (no chained dependency on level 0's full-resolution data),
    # then coarsen by 2x and write.
    for i in range(1, n_levels):
        enc = pyramid.encoding[f"/{i}"]
        template_ds = pyramid.dt[f"/{i}"].ds

        # Opening from zarr breaks the dask graph chain.
        prev_template = pyramid.dt[f"/{i - 1}"].ds
        prev_enc = pyramid.encoding[f"/{i - 1}"]
        prev_ds = _open_level(store, i - 1, encoding=prev_enc, template=prev_template)

        curr_ds = getattr(
            prev_ds.coarsen({x_dim: 2, y_dim: 2}, boundary="trim"), method
        )()

        # coarsen drops dataset and variable attributes; restore from template.
        curr_ds.attrs = template_ds.attrs
        for var in curr_ds.data_vars:
            if var in template_ds:
                curr_ds[var].attrs = template_ds[var].attrs

        _print_level_start(i, n_levels, template_ds, x_dim, y_dim)
        t0 = time.perf_counter()
        _write_level(curr_ds, store, i, enc, zarr_format)
        _print_level_done(i, n_levels, time.perf_counter() - t0)

    if consolidated and zarr_format != 3:
        zarr.consolidate_metadata(store)
