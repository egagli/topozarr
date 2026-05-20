import numpy as np
import pytest
import zarr
import zarr.storage
from zarr.storage._fsspec import _make_async
import xarray as xr
from topozarr.coarsen import create_pyramid
from topozarr.write import write_pyramid


def _mem_fs():
    fsspec = pytest.importorskip("fsspec")
    return fsspec.filesystem("memory")


def _fss(fs, path):
    """Build a FsspecStore, wrapping sync filesystems as needed."""
    return zarr.storage.FsspecStore(_make_async(fs), path=path)


def test_write_pyramid_structure(create_dataset, tmp_path):
    """write_pyramid writes all levels and the root multiscale attrs."""
    import fsspec
    fs = fsspec.filesystem("memory")
    path = "/test/structure"
    ds = create_dataset(nx=32, ny=32)
    pyramid = create_pyramid(ds, levels=3)

    write_pyramid(pyramid, fs, path, zarr_format=3)

    root = zarr.open_group(_fss(fs, path), mode="r")
    assert "multiscales" in dict(root.attrs)
    assert set(root.group_keys()) == {"0", "1", "2"}


def test_write_pyramid_shapes(create_dataset):
    """Each level's on-disk shape matches the pyramid's DataTree shape."""
    import fsspec
    fs = fsspec.filesystem("memory")
    path = "/test/shapes"
    ds = create_dataset(nx=32, ny=32)
    pyramid = create_pyramid(ds, levels=3)

    write_pyramid(pyramid, fs, path, zarr_format=3)

    for i in range(3):
        level_ds = xr.open_zarr(_fss(fs, f"{path}/{i}"), consolidated=False)
        expected_shape = pyramid.dt[f"/{i}"].ds["elevation"].shape
        assert level_ds["elevation"].shape == expected_shape, (
            f"Level {i} shape mismatch: got {level_ds['elevation'].shape}, "
            f"expected {expected_shape}"
        )


def test_write_pyramid_values_consistent(create_dataset):
    """Values written by write_pyramid are close to the chained-coarsen values."""
    import fsspec
    fs = fsspec.filesystem("memory")
    path = "/test/values"
    ds = create_dataset(nx=32, ny=32)
    pyramid = create_pyramid(ds, levels=3)

    write_pyramid(pyramid, fs, path, zarr_format=3)

    written = xr.open_zarr(_fss(fs, f"{path}/2"), consolidated=False)["elevation"].values
    expected = pyramid.dt["/2"].ds["elevation"].compute().values
    np.testing.assert_allclose(written, expected, rtol=1e-5)


def test_write_pyramid_root_attrs(create_dataset):
    """Root-level spatial and multiscale attributes are written into zarr.json."""
    import fsspec
    fs = fsspec.filesystem("memory")
    path = "/test/root_attrs"
    ds = create_dataset(nx=16, ny=16)
    pyramid = create_pyramid(ds, levels=2)

    write_pyramid(pyramid, fs, path, zarr_format=3)

    root_attrs = dict(zarr.open_group(_fss(fs, path), mode="r").attrs)
    assert "spatial:transform" in root_attrs
    assert "spatial:bbox" in root_attrs
    assert "proj:code" in root_attrs
    assert "multiscales" in root_attrs


def test_write_pyramid_level_attrs(create_dataset):
    """Dataset-level attributes are preserved through write_pyramid."""
    import fsspec
    fs = fsspec.filesystem("memory")
    path = "/test/level_attrs"
    ds = create_dataset(nx=16, ny=16)
    ds.attrs["title"] = "test dataset"
    pyramid = create_pyramid(ds, levels=2)

    write_pyramid(pyramid, fs, path, zarr_format=3)

    for i in range(2):
        level_ds = xr.open_zarr(_fss(fs, f"{path}/{i}"), consolidated=False)
        assert level_ds.attrs.get("title") == "test dataset"


def test_write_pyramid_encoding(create_dataset):
    """Encoding (dtype override) is applied when writing."""
    import fsspec
    fs = fsspec.filesystem("memory")
    path = "/test/encoding"
    ds = create_dataset(nx=32, ny=32)
    pyramid = create_pyramid(ds, levels=2)

    for level_enc in pyramid.encoding.values():
        for var_enc in level_enc.values():
            var_enc["dtype"] = "int16"
            var_enc["_FillValue"] = np.iinfo(np.int16).min

    write_pyramid(pyramid, fs, path, zarr_format=3)

    # xarray decodes int16+_FillValue back to float on read; check zarr directly
    z = zarr.open_group(_fss(fs, path), mode="r")
    assert z["0"]["elevation"].dtype == np.int16


def test_write_pyramid_memory_fs(create_dataset):
    """write_pyramid works correctly with an in-memory fsspec filesystem.

    Uses fsspec's memory filesystem as a stand-in for cloud stores
    (Azure, GCS) — same FsspecStore interface, no credentials required.
    """
    fs = _mem_fs()
    path = "/test/pyramid"
    ds = create_dataset(nx=32, ny=32)
    pyramid = create_pyramid(ds, levels=3)

    write_pyramid(pyramid, fs, path, zarr_format=3)

    root_attrs = dict(zarr.open_group(_fss(fs, path), mode="r").attrs)
    assert "multiscales" in root_attrs
    assert "proj:code" in root_attrs

    for i in range(3):
        level_ds = xr.open_zarr(_fss(fs, f"{path}/{i}"), consolidated=False)
        expected = pyramid.dt[f"/{i}"].ds["elevation"].shape
        assert level_ds["elevation"].shape == expected, (
            f"Memory FS level {i}: got {level_ds['elevation'].shape}, expected {expected}"
        )


def test_write_pyramid_custom_dims(create_dataset):
    """write_pyramid respects custom x_dim/y_dim/method arguments."""
    import fsspec
    fs = fsspec.filesystem("memory")
    path = "/test/custom_dims"
    ds = create_dataset(x_dim="lon", y_dim="lat")
    pyramid = create_pyramid(ds, levels=2, x_dim="lon", y_dim="lat", method="max")

    write_pyramid(pyramid, fs, path, x_dim="lon", y_dim="lat", method="max", zarr_format=3)

    root = zarr.open_group(_fss(fs, path), mode="r")
    assert set(root.group_keys()) == {"0", "1"}
