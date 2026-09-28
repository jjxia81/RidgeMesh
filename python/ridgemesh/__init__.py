"""Python package for RidgeMesh's native height-ridge surfacer."""

from .ridge_surface import (
    Bounds3D,
    LongestEdgeRefinementOptions,
    LinearitySampling,
    Polygon,
    PolygonTriangulation,
    RefinementPipeline,
    RidgeCurvatureFilter,
    RefinementTarget,
    SurfaceMesh,
    SurfaceOptions,
    Triangle,
    Vec3,
    extract_height_ridges,
    extract_height_ridges_from_derivatives,
    directional_linearity_error_16,
    directional_linearity_error_5,
)
from .torch_adapter import TorchFieldAdapter, extract_torch_udf

__all__ = [
    "Bounds3D",
    "LongestEdgeRefinementOptions",
    "LinearitySampling",
    "Polygon",
    "PolygonTriangulation",
    "RefinementPipeline",
    "RidgeCurvatureFilter",
    "RefinementTarget",
    "SurfaceMesh",
    "SurfaceOptions",
    "TorchFieldAdapter",
    "Triangle",
    "Vec3",
    "extract_height_ridges",
    "extract_height_ridges_from_derivatives",
    "directional_linearity_error_16",
    "directional_linearity_error_5",
    "extract_torch_udf",
]
