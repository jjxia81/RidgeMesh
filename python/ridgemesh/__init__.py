"""Python package for RidgeMesh's native height-ridge surfacer."""

from .ridge_surface import (
    Bounds3D,
    LongestEdgeRefinementOptions,
    RefinementTarget,
    SurfaceMesh,
    SurfaceOptions,
    Triangle,
    Vec3,
    extract_height_ridges,
    extract_height_ridges_from_derivatives,
)
from .torch_adapter import TorchFieldAdapter, extract_torch_udf

__all__ = [
    "Bounds3D",
    "LongestEdgeRefinementOptions",
    "RefinementTarget",
    "SurfaceMesh",
    "SurfaceOptions",
    "TorchFieldAdapter",
    "Triangle",
    "Vec3",
    "extract_height_ridges",
    "extract_height_ridges_from_derivatives",
    "extract_torch_udf",
]
