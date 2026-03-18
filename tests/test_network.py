#!/usr/bin/env python

"""Tests for `geoai.network` module."""

import inspect
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Polygon


def _create_mask_raster(path, width=100, height=100):
    """Create a binary mask raster with a cross pattern (projected CRS)."""
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


# -----------------------------------------------------------------------
# neatify_network
# -----------------------------------------------------------------------


class TestNeatifyNetworkSignature(unittest.TestCase):
    """Tests for the neatify_network function signature."""

    def test_function_exists(self):
        """Test that neatify_network is importable from geoai.network."""
        from geoai.network import neatify_network

        self.assertTrue(callable(neatify_network))

    def test_function_signature(self):
        """Test that neatify_network has expected parameters."""
        from geoai.network import neatify_network

        sig = inspect.signature(neatify_network)
        self.assertIn("gdf", sig.parameters)


class TestNeatifyNetworkValidation(unittest.TestCase):
    """Tests for neatify_network input validation."""

    def test_empty_gdf_returns_copy(self):
        """Test that an empty GeoDataFrame returns an empty copy."""
        from geoai.network import neatify_network

        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:32618")
        result = neatify_network(gdf)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 0)

    def test_rejects_polygon_geometries(self):
        """Test that Polygon geometries raise ValueError."""
        from geoai.network import neatify_network

        gdf = gpd.GeoDataFrame(
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:32618",
        )
        with self.assertRaises(ValueError) as ctx:
            neatify_network(gdf)
        self.assertIn("Polygon", str(ctx.exception))

    def test_rejects_geographic_crs(self):
        """Test that geographic CRS raises ValueError."""
        from geoai.network import neatify_network

        gdf = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 1)])],
            crs="EPSG:4326",
        )
        with self.assertRaises(ValueError) as ctx:
            neatify_network(gdf)
        self.assertIn("geographic", str(ctx.exception).lower())

    def test_rejects_no_crs(self):
        """Test that missing CRS raises ValueError."""
        from geoai.network import neatify_network

        gdf = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])])
        with self.assertRaises(ValueError) as ctx:
            neatify_network(gdf)
        self.assertIn("no CRS", str(ctx.exception))

    @patch.dict("sys.modules", {"neatnet": MagicMock()})
    def test_accepts_linestring_projected(self):
        """Test that valid LineString GeoDataFrame with projected CRS is accepted."""
        import sys

        from geoai.network import neatify_network

        mock_neatnet = sys.modules["neatnet"]
        expected = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 1)])],
            crs="EPSG:32618",
        )
        mock_neatnet.neatify.return_value = expected

        gdf = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 1)])],
            crs="EPSG:32618",
        )
        result = neatify_network(gdf)
        mock_neatnet.neatify.assert_called_once()
        self.assertIsInstance(result, gpd.GeoDataFrame)

    def test_import_error_without_neatnet(self):
        """Test that ImportError is raised when neatnet is not installed."""
        from geoai.network import neatify_network

        gdf = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 1)])],
            crs="EPSG:32618",
        )
        with patch.dict("sys.modules", {"neatnet": None}):
            with self.assertRaises(ImportError) as ctx:
                neatify_network(gdf)
            self.assertIn("neatnet", str(ctx.exception))


# -----------------------------------------------------------------------
# Topology utilities
# -----------------------------------------------------------------------


class TestTopologyUtilities(unittest.TestCase):
    """Tests for close_gaps, extend_lines, fix_topology wrappers."""

    def _make_line_gdf(self):
        return gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (100, 0)]), LineString([(0, 50), (100, 50)])],
            crs="EPSG:32618",
        )

    def test_close_gaps_signature(self):
        """Test that close_gaps has expected parameters."""
        from geoai.network import close_gaps

        sig = inspect.signature(close_gaps)
        self.assertIn("gdf", sig.parameters)
        self.assertIn("tolerance", sig.parameters)

    def test_extend_lines_signature(self):
        """Test that extend_lines has expected parameters."""
        from geoai.network import extend_lines

        sig = inspect.signature(extend_lines)
        self.assertIn("gdf", sig.parameters)
        self.assertIn("tolerance", sig.parameters)

    def test_fix_topology_signature(self):
        """Test that fix_topology has expected parameters."""
        from geoai.network import fix_topology

        sig = inspect.signature(fix_topology)
        self.assertIn("gdf", sig.parameters)

    def test_close_gaps_empty_returns_copy(self):
        """Test that empty GeoDataFrame returns empty copy."""
        from geoai.network import close_gaps

        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:32618")
        result = close_gaps(gdf, tolerance=5.0)
        self.assertEqual(len(result), 0)

    def test_extend_lines_empty_returns_copy(self):
        """Test that empty GeoDataFrame returns empty copy."""
        from geoai.network import extend_lines

        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:32618")
        result = extend_lines(gdf, tolerance=5.0)
        self.assertEqual(len(result), 0)

    def test_fix_topology_empty_returns_copy(self):
        """Test that empty GeoDataFrame returns empty copy."""
        from geoai.network import fix_topology

        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:32618")
        result = fix_topology(gdf)
        self.assertEqual(len(result), 0)

    @patch.dict("sys.modules", {"neatnet": MagicMock()})
    def test_close_gaps_calls_neatnet(self):
        """Test that close_gaps delegates to neatnet.close_gaps."""
        import sys

        from geoai.network import close_gaps

        mock = sys.modules["neatnet"]
        mock.close_gaps.return_value = self._make_line_gdf()

        close_gaps(self._make_line_gdf(), tolerance=5.0)
        mock.close_gaps.assert_called_once()

    @patch.dict("sys.modules", {"neatnet": MagicMock()})
    def test_extend_lines_calls_neatnet(self):
        """Test that extend_lines delegates to neatnet.extend_lines."""
        import sys

        from geoai.network import extend_lines

        mock = sys.modules["neatnet"]
        mock.extend_lines.return_value = self._make_line_gdf()

        extend_lines(self._make_line_gdf(), tolerance=10.0)
        mock.extend_lines.assert_called_once()

    @patch.dict("sys.modules", {"neatnet": MagicMock()})
    def test_fix_topology_calls_neatnet(self):
        """Test that fix_topology delegates to neatnet.fix_topology."""
        import sys

        from geoai.network import fix_topology

        mock = sys.modules["neatnet"]
        mock.fix_topology.return_value = self._make_line_gdf()

        fix_topology(self._make_line_gdf())
        mock.fix_topology.assert_called_once()

    def test_topology_rejects_polygons(self):
        """Test that topology functions reject non-line geometries."""
        from geoai.network import close_gaps, extend_lines, fix_topology

        gdf = gpd.GeoDataFrame(
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:32618",
        )
        with self.assertRaises(ValueError):
            close_gaps(gdf, tolerance=5.0)
        with self.assertRaises(ValueError):
            extend_lines(gdf, tolerance=5.0)
        with self.assertRaises(ValueError):
            fix_topology(gdf)


# -----------------------------------------------------------------------
# extract_line_network
# -----------------------------------------------------------------------


class TestExtractLineNetworkSignature(unittest.TestCase):
    """Tests for the extract_line_network function signature."""

    def test_function_exists(self):
        """Test that extract_line_network is importable."""
        from geoai.network import extract_line_network

        self.assertTrue(callable(extract_line_network))

    def test_function_signature(self):
        """Test that extract_line_network has expected parameters."""
        from geoai.network import extract_line_network

        sig = inspect.signature(extract_line_network)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("neatify", sig.parameters)

    def test_neatify_defaults_false(self):
        """Test that neatify defaults to False for extract_line_network."""
        from geoai.network import extract_line_network

        sig = inspect.signature(extract_line_network)
        self.assertFalse(sig.parameters["neatify"].default)


class TestExtractLineNetwork(unittest.TestCase):
    """Integration tests for extract_line_network."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mask_path = os.path.join(self.tmpdir, "mask.tif")
        _create_mask_raster(self.mask_path)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_geodataframe(self):
        """Test that result is a GeoDataFrame."""
        from geoai.network import extract_line_network

        gdf = extract_line_network(self.mask_path)
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertGreater(len(gdf), 0)

    def test_neatify_false_by_default(self):
        """Test that neatify=False works without neatnet."""
        from geoai.network import extract_line_network

        gdf = extract_line_network(self.mask_path)
        self.assertIsInstance(gdf, gpd.GeoDataFrame)


# -----------------------------------------------------------------------
# extract_road_network
# -----------------------------------------------------------------------


class TestExtractRoadNetworkSignature(unittest.TestCase):
    """Tests for the extract_road_network function signature."""

    def test_function_exists(self):
        """Test that extract_road_network is importable from geoai.network."""
        from geoai.network import extract_road_network

        self.assertTrue(callable(extract_road_network))

    def test_neatify_defaults_true(self):
        """Test that neatify defaults to True for extract_road_network."""
        from geoai.network import extract_road_network

        sig = inspect.signature(extract_road_network)
        self.assertTrue(sig.parameters["neatify"].default)


class TestExtractRoadNetwork(unittest.TestCase):
    """Integration tests for extract_road_network with synthetic rasters."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mask_path = os.path.join(self.tmpdir, "road_mask.tif")
        _create_mask_raster(self.mask_path)

        self.empty_mask_path = os.path.join(self.tmpdir, "empty_mask.tif")
        _create_mask_raster(self.empty_mask_path, width=10, height=10)
        with rasterio.open(self.empty_mask_path, "r+") as dst:
            dst.write(np.zeros((1, 10, 10), dtype=np.uint8))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_geodataframe(self):
        from geoai.network import extract_road_network

        gdf = extract_road_network(self.mask_path, neatify=False)
        self.assertIsInstance(gdf, gpd.GeoDataFrame)

    def test_output_has_linestrings(self):
        from geoai.network import extract_road_network

        gdf = extract_road_network(self.mask_path, neatify=False)
        self.assertGreater(len(gdf), 0)
        for geom_type in gdf.geom_type.unique():
            self.assertIn(geom_type, {"LineString", "MultiLineString"})

    def test_output_has_crs(self):
        from geoai.network import extract_road_network

        gdf = extract_road_network(self.mask_path, neatify=False)
        self.assertEqual(gdf.crs.to_epsg(), 32618)

    def test_saves_to_gpkg(self):
        from geoai.network import extract_road_network

        output_path = os.path.join(self.tmpdir, "roads.gpkg")
        gdf = extract_road_network(
            self.mask_path, output_path=output_path, neatify=False
        )
        self.assertTrue(os.path.exists(output_path))
        loaded = gpd.read_file(output_path)
        self.assertEqual(len(loaded), len(gdf))

    def test_saves_to_geojson(self):
        from geoai.network import extract_road_network

        output_path = os.path.join(self.tmpdir, "roads.geojson")
        extract_road_network(self.mask_path, output_path=output_path, neatify=False)
        self.assertTrue(os.path.exists(output_path))

    def test_min_length_filters(self):
        from geoai.network import extract_road_network

        gdf_all = extract_road_network(self.mask_path, neatify=False, min_length=0)
        gdf_filtered = extract_road_network(
            self.mask_path, neatify=False, min_length=50.0
        )
        self.assertLessEqual(len(gdf_filtered), len(gdf_all))

    def test_empty_mask_returns_empty(self):
        from geoai.network import extract_road_network

        gdf = extract_road_network(self.empty_mask_path, neatify=False)
        self.assertEqual(len(gdf), 0)


# -----------------------------------------------------------------------
# Backward compatibility: geoai.road shim
# -----------------------------------------------------------------------


class TestBackwardCompat(unittest.TestCase):
    """Test that imports from geoai.road still work."""

    def test_import_neatify_network(self):
        from geoai.road import neatify_network

        self.assertTrue(callable(neatify_network))

    def test_import_extract_road_network(self):
        from geoai.road import extract_road_network

        self.assertTrue(callable(extract_road_network))

    def test_import_extract_line_network(self):
        from geoai.road import extract_line_network

        self.assertTrue(callable(extract_line_network))


# -----------------------------------------------------------------------
# polygons_to_line_network
# -----------------------------------------------------------------------


class TestPolygonsToLineNetwork(unittest.TestCase):
    """Tests for polygons_to_line_network utility."""

    def _make_adjacent_polygons(self):
        """Create two adjacent square polygons sharing an edge."""
        p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        p2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])
        return gpd.GeoDataFrame(geometry=[p1, p2], crs="EPSG:32618")

    def test_returns_geodataframe(self):
        from geoai.utils.vector import polygons_to_line_network

        gdf = self._make_adjacent_polygons()
        result = polygons_to_line_network(gdf)
        self.assertIsInstance(result, gpd.GeoDataFrame)

    def test_output_has_linestrings(self):
        from geoai.utils.vector import polygons_to_line_network

        result = polygons_to_line_network(self._make_adjacent_polygons())
        self.assertGreater(len(result), 0)
        for geom_type in result.geom_type.unique():
            self.assertIn(geom_type, {"LineString", "MultiLineString"})

    def test_crs_preserved(self):
        from geoai.utils.vector import polygons_to_line_network

        result = polygons_to_line_network(self._make_adjacent_polygons())
        self.assertEqual(result.crs.to_epsg(), 32618)

    def test_merge_deduplicates_shared_edges(self):
        """Test that merge=True deduplicates shared boundary edges."""
        from geoai.utils.vector import polygons_to_line_network

        gdf = self._make_adjacent_polygons()
        merged = polygons_to_line_network(gdf, merge=True)
        # Two adjacent squares share one edge; merged should produce
        # a connected boundary network (not duplicate the shared edge)
        total_merged_length = merged.geometry.length.sum()

        unmerged = polygons_to_line_network(gdf, merge=False)
        total_unmerged_length = unmerged.geometry.length.sum()

        # Merged total length should be less (shared edge counted once)
        self.assertLess(total_merged_length, total_unmerged_length)

    def test_min_length_filters(self):
        from geoai.utils.vector import polygons_to_line_network

        gdf = self._make_adjacent_polygons()
        all_lines = polygons_to_line_network(gdf, min_length=0)
        filtered = polygons_to_line_network(gdf, min_length=15.0)
        self.assertLessEqual(len(filtered), len(all_lines))

    def test_empty_gdf(self):
        from geoai.utils.vector import polygons_to_line_network

        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:32618")
        result = polygons_to_line_network(gdf)
        self.assertEqual(len(result), 0)

    def test_saves_output(self):
        from geoai.utils.vector import polygons_to_line_network

        tmpdir = tempfile.mkdtemp()
        try:
            output_path = os.path.join(tmpdir, "lines.gpkg")
            polygons_to_line_network(
                self._make_adjacent_polygons(), output_path=output_path
            )
            self.assertTrue(os.path.exists(output_path))
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
