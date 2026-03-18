"""Backward-compatibility shim — use ``geoai.network`` instead.

All functions have moved to :mod:`geoai.network`. This module re-exports
them so that existing ``from geoai.road import ...`` imports continue to
work.
"""

from geoai.network import (  # noqa: F401
    extract_line_network,
    extract_road_network,
    neatify_network,
)
