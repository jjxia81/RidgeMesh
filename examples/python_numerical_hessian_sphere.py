"""Smoke-test RidgeMesh's PyTorch finite-difference derivative path.

The scalar field has a height ridge at ``||x|| = radius``.  The test asks
``extract_torch_udf`` to estimate *both* its gradient and Hessian from scalar
function values, then writes the extracted ridge triangles to a PLY file.

Run after installing the Python package (including the optional torch extra):

    python -m pip install ".[torch]"
    python examples/python_numerical_hessian_sphere.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ridgemesh as rm


def write_ply(mesh: rm.SurfaceMesh, path: Path) -> None:
    """Write only ridge triangles as an ASCII PLY mesh."""
    faces = mesh.ridge_triangles
    with path.open("w", encoding="ascii") as output:
        output.write("ply\nformat ascii 1.0\n")
        output.write(f"element vertex {len(mesh.vertices)}\n")
        output.write("property float x\nproperty float y\nproperty float z\n")
        output.write(f"element face {len(faces)}\n")
        output.write("property list uchar int vertex_indices\nend_header\n")
        for vertex in mesh.vertices:
            output.write(f"{vertex.x:.9g} {vertex.y:.9g} {vertex.z:.9g}\n")
        for triangle in faces:
            first, second, third = triangle.indices
            output.write(f"3 {first} {second} {third}\n")


def ridge_radius_errors(mesh: rm.SurfaceMesh, expected_radius: float):
    """Return min/max/mean absolute radial error of ridge vertices."""
    import math

    errors = [
        abs(math.sqrt(vertex.x**2 + vertex.y**2 + vertex.z**2) - expected_radius)
        for vertex in mesh.vertices
    ]
    if not errors:
        raise AssertionError("extraction returned no ridge vertices")
    return min(errors), max(errors), sum(errors) / len(errors)


class SphereRidge:
    """A GeoUDF-shaped callable whose ridge is a sphere of a known radius."""

    def __init__(self, radius: float):
        self.radius = radius

    def __call__(self, _context, query):
        # query has the GeoUDF convention: (batch, xyz, point_count).
        radius_squared = (query * query).sum(dim=1, keepdim=True)
        offset = radius_squared - self.radius * self.radius
        return -(offset * offset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=20, help="initial uniform grid resolution per axis")
    parser.add_argument("--radius", type=float, default=0.6, help="known sphere radius")
    parser.add_argument("--step", type=float, default=5e-3, help="central finite-difference step")
    parser.add_argument("--output", type=Path, default=Path("numerical_hessian_sphere.ply"))
    arguments = parser.parse_args()

    if arguments.resolution < 2:
        parser.error("--resolution must be at least 2")
    if arguments.step <= 0.0:
        parser.error("--step must be positive")

    options = rm.SurfaceOptions()
    options.nx = options.ny = options.nz = arguments.resolution
    options.surface_target = rm.RefinementTarget.ridges
    options.longest_edge_refinement.target = rm.RefinementTarget.none
    options.longest_edge_refinement.max_splits = 0  # A reproducible uniform-grid test.
    options.root_iterations = 6

    bounds = rm.Bounds3D(rm.Vec3(-1.0, -1.0, -1.0), rm.Vec3(1.0, 1.0, 1.0))
    mesh = rm.extract_torch_udf(
        SphereRidge(arguments.radius),
        context={},
        bounds=bounds,
        options=options,
        device="cpu",
        derivative_mode="finite_difference",
        finite_difference_step=arguments.step,
    )

    if not mesh.ridge_triangles:
        raise AssertionError("numerical-Hessian extraction returned no ridge triangles")

    minimum, maximum, mean = ridge_radius_errors(mesh, arguments.radius)
    # The 20^3 test grid has a 0.1 cell size.  Root refinement makes the
    # expected radial error much smaller; this tolerant bound guards against
    # real regressions without depending on a particular compiler or torch build.
    if maximum > 0.03:
        raise AssertionError(f"sphere ridge is inaccurate: maximum radial error is {maximum:.6g}")

    write_ply(mesh, arguments.output)
    print(f"ridge triangles: {len(mesh.ridge_triangles)}")
    print(f"radial error (min / mean / max): {minimum:.6g} / {mean:.6g} / {maximum:.6g}")
    print(f"wrote {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
