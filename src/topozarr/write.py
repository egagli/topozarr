from __future__ import annotations
from typing import Any
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


def _open_level(store: Any, level: int) -> xr.Dataset:
    if _is_fsmap(store):
        # zarr v3 raises ValueError if group/path is passed alongside an FSMap.
        # Opening the sub-mapper as a zarr Group first converts it to a native
        # FsspecStore, which xr.open_zarr accepts without the restriction.
        sub = _sub_mapper(store, level)
        zarr_group = zarr.open_group(sub, mode="r")
        return xr.open_zarr(zarr_group, consolidated=False)
    else:
        return xr.open_zarr(store, group=str(level), consolidated=False)


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
    _write_level(pyramid.dt["/0"].ds, store, 0, pyramid.encoding["/0"], zarr_format)

    # For each subsequent level, read the previous level back from zarr to get
    # fresh dask arrays (no chained dependency on level 0's full-resolution data),
    # then coarsen by 2x and write.
    for i in range(1, n_levels):
        enc = pyramid.encoding[f"/{i}"]
        template_ds = pyramid.dt[f"/{i}"].ds

        # Opening from zarr breaks the dask graph chain.
        prev_ds = _open_level(store, i - 1)

        curr_ds = getattr(
            prev_ds.coarsen({x_dim: 2, y_dim: 2}, boundary="trim"), method
        )()

        # coarsen drops dataset and variable attributes; restore from template.
        curr_ds.attrs = template_ds.attrs
        for var in curr_ds.data_vars:
            if var in template_ds:
                curr_ds[var].attrs = template_ds[var].attrs

        _write_level(curr_ds, store, i, enc, zarr_format)

    if consolidated and zarr_format != 3:
        zarr.consolidate_metadata(store)
