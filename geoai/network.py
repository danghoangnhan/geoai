"""Line network extraction, topology, and post-processing.

This module provides tools for extracting line networks from deep learning
segmentation masks and cleaning them using the neatnet library. It supports
any linear feature domain — roads, rivers, field boundaries, pipelines, etc.

The core pipeline converts raster masks into clean vector LineString networks
by skeletonizing the mask, vectorizing the skeleton, and optionally simplifying
the network with neatnet's adaptive continuity-preserving simplification.

Individual topology utilities (``close_gaps``, ``extend_lines``,
``fix_topology``) are also exposed for fine-grained network cleanup.

Reference:
    https://github.com/uscuni/neatnet
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import geopandas as gpd


def _validate_line_gdf(gdf: gpd.GeoDataFrame, require_projected: bool = True) -> None:
    """Validate that a GeoDataFrame contains line geometries in a projected CRS.

    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame to validate.
        require_projected (bool): Whether to require a projected CRS.

    Raises:
        ValueError: If validation fails.
    """
    if len(gdf) == 0:
        return

    geom_types = gdf.geom_type.unique()
    valid_types = {"LineString", "MultiLineString"}
    invalid = set(geom_types) - valid_types
    if invalid:
        raise ValueError(
            f"Expected LineString or MultiLineString geometries, "
            f"but found: {invalid}. Convert your geometries to lines first."
        )

    if require_projected:
        if gdf.crs is None:
            raise ValueError(
                "GeoDataFrame has no CRS. A projected CRS is required "
                "(e.g., UTM). Set the CRS with gdf.set_crs() or gdf.to_crs()."
            )
        if gdf.crs.is_geographic:
            raise ValueError(
                f"GeoDataFrame uses a geographic CRS ({gdf.crs}). "
                "A projected CRS (e.g., UTM) with units in meters is "
                "required. Reproject with gdf.to_crs() first."
            )


def _import_neatnet():
    """Import neatnet, raising a helpful error if not installed."""
    try:
        import neatnet

        return neatnet
    except ImportError:
        raise ImportError(
            "neatnet is required for network topology operations. "
            "Install it with: pip install neatnet\n"
            "Or install geoai with network support: pip install geoai-py[network]\n"
            "Note: neatnet requires Python >= 3.11."
        )


def _save_gdf(gdf: gpd.GeoDataFrame, output_path: str) -> None:
    """Save a GeoDataFrame to file, auto-detecting format from extension."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".geojson":
        gdf.to_file(output_path, driver="GeoJSON")
    elif ext == ".shp":
        gdf.to_file(output_path, driver="ESRI Shapefile")
    else:
        gdf.to_file(output_path, driver="GPKG")


# -----------------------------------------------------------------------
# Core network simplification
# -----------------------------------------------------------------------


def neatify_network(
    gdf: gpd.GeoDataFrame,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Simplify and clean a line network GeoDataFrame using neatnet.

    Wrapper around ``neatnet.neatify()`` that validates input geometry types
    and coordinate reference system before processing. Works best on street
    networks but can also clean river networks, field boundary networks, and
    other planar line graphs.

    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame of LineString geometries
            in a projected coordinate reference system (not geographic).
        **kwargs: Additional keyword arguments passed to ``neatnet.neatify()``.
            See the neatnet documentation for available options.

    Returns:
        geopandas.GeoDataFrame: Simplified GeoDataFrame of LineString
            geometries with artifacts (dual carriageways, roundabouts,
            complex intersections) replaced by centerlines.

    Raises:
        ImportError: If neatnet is not installed (requires Python >= 3.11).
        ValueError: If the GeoDataFrame has no LineString geometries or uses
            a geographic (lat/lon) CRS instead of a projected CRS.

    Example:
        >>> import geopandas as gpd
        >>> from geoai.network import neatify_network
        >>> roads = gpd.read_file("road_lines.gpkg")
        >>> cleaned = neatify_network(roads)
    """
    if len(gdf) == 0:
        return gdf.copy()

    _validate_line_gdf(gdf)
    neatnet = _import_neatnet()
    return neatnet.neatify(gdf, **kwargs)


# -----------------------------------------------------------------------
# Topology utilities
# -----------------------------------------------------------------------


def close_gaps(
    gdf: gpd.GeoDataFrame,
    tolerance: float,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Snap disconnected line endpoints within a tolerance.

    Closes small gaps in a line network by snapping endpoints that are
    within ``tolerance`` distance of another line. Useful for fixing
    fragmented networks from noisy segmentation results.

    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame of LineString geometries
            in a projected CRS.
        tolerance (float): Maximum distance (in map units) for gap closure.
        **kwargs: Additional keyword arguments passed to
            ``neatnet.close_gaps()``.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with gaps closed.

    Raises:
        ImportError: If neatnet is not installed.
        ValueError: If input is invalid.

    Example:
        >>> from geoai.network import close_gaps
        >>> fixed = close_gaps(fragmented_lines, tolerance=5.0)
    """
    if len(gdf) == 0:
        return gdf.copy()

    _validate_line_gdf(gdf)
    neatnet = _import_neatnet()
    return neatnet.close_gaps(gdf, tolerance, **kwargs)


def extend_lines(
    gdf: gpd.GeoDataFrame,
    tolerance: float,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Extend line endpoints to connect nearby network components.

    Extends dangling line endpoints along their direction until they
    intersect another line or reach the ``tolerance`` distance. Useful
    for connecting disconnected network components.

    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame of LineString geometries
            in a projected CRS.
        tolerance (float): Maximum extension distance (in map units).
        **kwargs: Additional keyword arguments passed to
            ``neatnet.extend_lines()``.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with lines extended.

    Raises:
        ImportError: If neatnet is not installed.
        ValueError: If input is invalid.

    Example:
        >>> from geoai.network import extend_lines
        >>> connected = extend_lines(road_lines, tolerance=10.0)
    """
    if len(gdf) == 0:
        return gdf.copy()

    _validate_line_gdf(gdf)
    neatnet = _import_neatnet()
    return neatnet.extend_lines(gdf, tolerance, **kwargs)


def fix_topology(
    gdf: gpd.GeoDataFrame,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Fix topological errors in a line network.

    Splits lines at intersections and removes duplicate segments to
    ensure a topologically correct planar graph. This is often a useful
    preprocessing step before ``neatify_network()``.

    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame of LineString geometries
            in a projected CRS.
        **kwargs: Additional keyword arguments passed to
            ``neatnet.fix_topology()``.

    Returns:
        geopandas.GeoDataFrame: Topologically corrected GeoDataFrame.

    Raises:
        ImportError: If neatnet is not installed.
        ValueError: If input is invalid.

    Example:
        >>> from geoai.network import fix_topology
        >>> clean = fix_topology(raw_lines)
    """
    if len(gdf) == 0:
        return gdf.copy()

    _validate_line_gdf(gdf)
    neatnet = _import_neatnet()
    return neatnet.fix_topology(gdf, **kwargs)


# -----------------------------------------------------------------------
# Network extraction from raster masks
# -----------------------------------------------------------------------


def extract_line_network(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    band: int = 1,
    threshold: float = 0,
    min_length: float = 0.0,
    simplify_tolerance: Optional[float] = None,
    neatify: bool = False,
    neatify_kwargs: Optional[Dict[str, Any]] = None,
) -> gpd.GeoDataFrame:
    """Extract a line network from any binary segmentation mask.

    Converts a binary mask raster (e.g., from a deep learning model) into
    a vector line network by:

    1. Skeletonizing the mask to extract centerlines
    2. Vectorizing the skeleton to LineString geometries
    3. Optionally simplifying the network with neatnet

    This is the generic version of ``extract_road_network()`` — it works
    for any linear feature (roads, rivers, pipelines, power lines, etc.).

    Args:
        input_path (str or Path): Path to the input binary mask raster
            (GeoTIFF). Pixel values greater than ``threshold`` are treated
            as foreground.
        output_path (str or Path, optional): Path to save the output vector
            file. The format is auto-detected from the file extension
            (.gpkg, .geojson, .shp). If None, returns the GeoDataFrame
            without saving.
        band (int): Band number to read from the raster (1-based).
            Defaults to 1.
        threshold (float): Pixel values greater than this threshold are
            treated as foreground. Defaults to 0.
        min_length (float): Minimum line length in map units to keep.
            Short lines are discarded as noise. Defaults to 0.0.
        simplify_tolerance (float, optional): Tolerance for Douglas-Peucker
            geometry simplification. None for no simplification.
        neatify (bool): Whether to apply neatnet post-processing.
            Requires neatnet to be installed. Defaults to False.
        neatify_kwargs (dict, optional): Additional keyword arguments
            passed to ``neatnet.neatify()``.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame of LineString geometries.

    Example:
        >>> import geoai
        >>> rivers = geoai.extract_line_network(
        ...     "water_mask.tif",
        ...     min_length=20.0,
        ...     neatify=False,
        ... )
    """
    from geoai.utils.raster import raster_mask_to_linestrings

    input_path = str(input_path)

    gdf = raster_mask_to_linestrings(
        raster_path=input_path,
        band=band,
        threshold=threshold,
        min_length=min_length,
        simplify_tolerance=simplify_tolerance,
    )

    if len(gdf) == 0:
        if output_path is not None:
            gdf.to_file(str(output_path), driver="GPKG")
        return gdf

    if neatify:
        kwargs = neatify_kwargs or {}
        gdf = neatify_network(gdf, **kwargs)

    if output_path is not None:
        _save_gdf(gdf, str(output_path))

    return gdf


def extract_road_network(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    band: int = 1,
    threshold: float = 0,
    min_length: float = 0.0,
    simplify_tolerance: Optional[float] = None,
    neatify: bool = True,
    neatify_kwargs: Optional[Dict[str, Any]] = None,
) -> gpd.GeoDataFrame:
    """Extract a clean road network from a road segmentation mask raster.

    Convenience wrapper around ``extract_line_network()`` with
    ``neatify=True`` by default, since road networks typically benefit
    from neatnet's simplification of dual carriageways and roundabouts.

    Args:
        input_path (str or Path): Path to the input road mask raster (GeoTIFF).
            Pixel values greater than ``threshold`` are treated as road.
        output_path (str or Path, optional): Path to save the output vector
            file. The format is auto-detected from the file extension
            (.gpkg, .geojson, .shp). If None, returns the GeoDataFrame
            without saving.
        band (int): Band number to read from the raster (1-based).
            Defaults to 1.
        threshold (float): Pixel values greater than this threshold are
            treated as road. Defaults to 0.
        min_length (float): Minimum line length in map units to keep.
            Short lines are discarded as noise. Defaults to 0.0.
        simplify_tolerance (float, optional): Tolerance for Douglas-Peucker
            geometry simplification applied before neatnet processing.
            None for no simplification.
        neatify (bool): Whether to apply neatnet post-processing to
            simplify the road network. Requires neatnet to be installed.
            Defaults to True.
        neatify_kwargs (dict, optional): Additional keyword arguments
            passed to ``neatnet.neatify()``.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame of LineString geometries
            representing the cleaned road network.

    Example:
        >>> import geoai
        >>> roads = geoai.extract_road_network(
        ...     "road_mask.tif",
        ...     output_path="roads.gpkg",
        ...     min_length=10.0,
        ... )
    """
    return extract_line_network(
        input_path=input_path,
        output_path=output_path,
        band=band,
        threshold=threshold,
        min_length=min_length,
        simplify_tolerance=simplify_tolerance,
        neatify=neatify,
        neatify_kwargs=neatify_kwargs,
    )
