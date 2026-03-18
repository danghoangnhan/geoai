#!/usr/bin/env python

"""Tests for raster_mask_to_linestrings and _skeleton_to_linestrings utilities."""

import os
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import Affine, from_bounds


def _create_cross_mask(path, width=100, height=100):
    """Create a binary raster with a cross pattern (projected CRS)."""
    data = np.zeros((1, height, width), dtype=np.uint8)
    data[0, height // 2, :] = 1
    data[0, :, width // 2] = 1

    transform = from_bounds(500000, 4500000, 500100, 4500100, width, height)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs="EPSG:32618",
        transform=transform,
    ) as dst:
        dst.write(data)


class TestRasterMaskToLinestrings(unittest.TestCase):
    """Tests for raster_mask_to_linestrings function."""

    def setUp(self):
        """Create temporary directory and test rasters."""
        self.tmpdir = tempfile.mkdtemp()

        self.cross_path = os.path.join(self.tmpdir, "cross.tif")
        _create_cross_mask(self.cross_path)

        self.empty_path = os.path.join(self.tmpdir, "empty.tif")
        _create_cross_mask(self.empty_path, width=20, height=20)
        with rasterio.open(self.empty_path, "r+") as dst:
            dst.write(np.zeros((1, 20, 20), dtype=np.uint8))

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_geodataframe(self):
        """Test that result is a GeoDataFrame."""
        from geoai.utils.raster import raster_mask_to_linestrings

        gdf = raster_mask_to_linestrings(self.cross_path)
        self.assertIsInstance(gdf, gpd.GeoDataFrame)

    def test_output_has_linestrings(self):
        """Test that output contains LineString geometries."""
        from geoai.utils.raster import raster_mask_to_linestrings

        gdf = raster_mask_to_linestrings(self.cross_path)
        self.assertGreater(len(gdf), 0)
        for geom_type in gdf.geom_type.unique():
            self.assertIn(geom_type, {"LineString", "MultiLineString"})

    def test_crs_matches_input(self):
        """Test that output CRS matches the input raster CRS."""
        from geoai.utils.raster import raster_mask_to_linestrings

        gdf = raster_mask_to_linestrings(self.cross_path)
        self.assertIsNotNone(gdf.crs)
        self.assertEqual(gdf.crs.to_epsg(), 32618)

    def test_threshold_filtering(self):
        """Test that only values > threshold are skeletonized."""
        from geoai.utils.raster import raster_mask_to_linestrings

        # With threshold=0 (default), values of 1 are included
        gdf_default = raster_mask_to_linestrings(self.cross_path, threshold=0)
        self.assertGreater(len(gdf_default), 0)

        # With threshold=1, values of 1 are excluded (need > 1)
        gdf_high = raster_mask_to_linestrings(self.cross_path, threshold=1)
        self.assertEqual(len(gdf_high), 0)

    def test_min_length_filtering(self):
        """Test that short segments are filtered by min_length."""
        from geoai.utils.raster import raster_mask_to_linestrings

        gdf_all = raster_mask_to_linestrings(self.cross_path, min_length=0)
        gdf_filtered = raster_mask_to_linestrings(self.cross_path, min_length=50.0)
        self.assertLessEqual(len(gdf_filtered), len(gdf_all))

    def test_simplify_tolerance(self):
        """Test that simplification is applied."""
        from geoai.utils.raster import raster_mask_to_linestrings

        gdf_raw = raster_mask_to_linestrings(self.cross_path)
        gdf_simplified = raster_mask_to_linestrings(
            self.cross_path, simplify_tolerance=5.0
        )
        # Both should produce results
        self.assertGreater(len(gdf_raw), 0)
        self.assertGreater(len(gdf_simplified), 0)

    def test_empty_mask(self):
        """Test that an all-zero mask returns an empty GeoDataFrame."""
        from geoai.utils.raster import raster_mask_to_linestrings

        gdf = raster_mask_to_linestrings(self.empty_path)
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertEqual(len(gdf), 0)

    def test_saves_gpkg(self):
        """Test that output is saved as GeoPackage."""
        from geoai.utils.raster import raster_mask_to_linestrings

        output_path = os.path.join(self.tmpdir, "lines.gpkg")
        gdf = raster_mask_to_linestrings(self.cross_path, output_path=output_path)
        self.assertTrue(os.path.exists(output_path))
        loaded = gpd.read_file(output_path)
        self.assertEqual(len(loaded), len(gdf))

    def test_saves_geojson(self):
        """Test that output is saved as GeoJSON."""
        from geoai.utils.raster import raster_mask_to_linestrings

        output_path = os.path.join(self.tmpdir, "lines.geojson")
        raster_mask_to_linestrings(self.cross_path, output_path=output_path)
        self.assertTrue(os.path.exists(output_path))


class TestSkeletonToLinestrings(unittest.TestCase):
    """Tests for the _skeleton_to_linestrings helper function."""

    def test_single_horizontal_line(self):
        """Test that a horizontal skeleton line produces a LineString."""
        from geoai.utils.raster import _skeleton_to_linestrings

        skeleton = np.zeros((10, 50), dtype=bool)
        skeleton[5, 5:45] = True
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)

        lines = _skeleton_to_linestrings(skeleton, transform)
        self.assertGreater(len(lines), 0)

        # All results should be LineStrings
        for line in lines:
            self.assertEqual(line.geom_type, "LineString")

    def test_cross_pattern(self):
        """Test that a cross pattern produces multiple LineStrings."""
        from geoai.utils.raster import _skeleton_to_linestrings

        skeleton = np.zeros((50, 50), dtype=bool)
        skeleton[25, :] = True
        skeleton[:, 25] = True
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 50.0)

        lines = _skeleton_to_linestrings(skeleton, transform)
        # A cross has a junction, so should produce multiple segments
        self.assertGreater(len(lines), 1)

    def test_empty_skeleton(self):
        """Test that an all-False skeleton returns an empty list."""
        from geoai.utils.raster import _skeleton_to_linestrings

        skeleton = np.zeros((10, 10), dtype=bool)
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)

        lines = _skeleton_to_linestrings(skeleton, transform)
        self.assertEqual(len(lines), 0)

    def test_coordinates_use_transform(self):
        """Test that output coordinates are in map space, not pixel space."""
        from geoai.utils.raster import _skeleton_to_linestrings

        skeleton = np.zeros((10, 10), dtype=bool)
        skeleton[5, 2:8] = True

        # Offset transform: origin at (1000, 2000), 10m pixels
        transform = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)

        lines = _skeleton_to_linestrings(skeleton, transform)
        self.assertGreater(len(lines), 0)

        # All coordinates should be in map space (around 1000-1100, 1900-2000)
        for line in lines:
            for x, y in line.coords:
                self.assertGreaterEqual(x, 1000.0)
                self.assertLessEqual(x, 1100.0)
                self.assertGreaterEqual(y, 1900.0)
                self.assertLessEqual(y, 2000.0)


if __name__ == "__main__":
    unittest.main()
