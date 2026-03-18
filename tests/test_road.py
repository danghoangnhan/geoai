#!/usr/bin/env python

"""Backward-compatibility tests — verify geoai.road shim still works.

The main test suite is in test_network.py. This file only verifies that
the ``geoai.road`` import path continues to resolve correctly.
"""

import unittest

import geopandas as gpd


class TestRoadShimImports(unittest.TestCase):
    """Test that all public symbols are importable from geoai.road."""

    def test_import_neatify_network(self):
        from geoai.road import neatify_network

        self.assertTrue(callable(neatify_network))

    def test_import_extract_road_network(self):
        from geoai.road import extract_road_network

        self.assertTrue(callable(extract_road_network))

    def test_import_extract_line_network(self):
        from geoai.road import extract_line_network

        self.assertTrue(callable(extract_line_network))

    def test_empty_gdf_via_shim(self):
        """Test that the shim delegates correctly."""
        from geoai.road import neatify_network

        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:32618")
        result = neatify_network(gdf)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
