# ridge_surface

A C++20 implementation of the 3-D strong height ridge / valley surfacer in
`height_ridge.nb`. It vendors MTet for tetrahedral connectivity and Eigen for
symmetric Hessian eigendecomposition. MTet, Eigen, and pybind11 are Git
submodules, so clone them together with the project; no system packages are
required:

```bash
git clone --recurse-submodules https://github.com/jjxia81/RidgeMesh.git
cd RidgeMesh
```

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/ridge_example
```

On this Windows machine, open PowerShell in this directory and run:

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\Release\ridge_example.exe
```

This uses the installed Visual Studio C++ compiler. The build produces
`build/Release/ridge_surface.lib` plus the example and test executables.

## Python

The default build also produces a native module named `ridge_surface` in the
build output directory. On Windows, after a Release build:

```powershell
$env:PYTHONPATH = "$(Resolve-Path .\vs2022\Release)"
python -c "import ridge_surface; print(ridge_surface.__doc__)"
```

```python
import ridge_surface as rs

box = rs.Bounds3D(rs.Vec3(-1, -1, -1), rs.Vec3(1, 1, 1))
options = rs.SurfaceOptions()
options.nx, options.ny, options.nz = 25, 24, 24

# Scalar callback: finite differences provide gradient and Hessian.
mesh = rs.extract_height_ridges(
    lambda x, y, z: -x*x - .1*y*y - .05*z*z,
    box,
    options,
)
print(len(mesh.vertices), len(mesh.ridge_triangles))
```

For accuracy and speed, use `extract_height_ridges_from_derivatives` with
Python callbacks returning a length-3 gradient and a 3-by-3 Hessian.

To inspect the scalar field itself, install `numpy` and `matplotlib`, then use
the slice helper copied next to the native module during the build:

```python
import ridge_surface_plot as rsp

f = lambda x, y, z: -x*x - .1*y*y - .05*z*z
figure, axes = rsp.plot_scalar_slice(f, bounds, axis="z", position=0.0)
figure.savefig("function_slice.png", dpi=180)

# Optional: plot the extracted ridge mesh in 3D.
figure, axes = rsp.plot_mesh(mesh)
figure.savefig("ridges.png", dpi=180)
```

The API accepts either a scalar field (central numerical derivatives are used)
or, preferably, exact gradient and Hessian callbacks:

```cpp
#include <ridge_surface/ridge_surface.hpp>
using namespace ridge_surface;

DifferentialField3D f{
  [](const Vec3& p) { return Vec3{-2*p.x, -2*p.y, -2*p.z}; },
  [](const Vec3&) { return Mat3{{{{-2,0,0}},{{0,-2,0}},{{0,0,-2}}}}; }
};
SurfaceOptions opt; opt.nx = opt.ny = opt.nz = 64;
SurfaceMesh result = extract_height_ridges(f, {{-1,-1,-1}, {1,1,1}}, opt);
```

## Adaptive longest-edge refinement

`nx`, `ny`, and `nz` define a coarse initial TET6 grid.  Enable the optional
MTet-backed adaptive stage to split the globally longest eligible edge before
surface extraction:

```cpp
SurfaceOptions opt;
opt.nx = opt.ny = opt.nz = 12; // coarse starting grid
opt.longest_edge_refinement.target = RefinementTarget::ridges_and_valleys;
opt.longest_edge_refinement.max_splits = 500;
opt.longest_edge_refinement.minimum_edge_length = 0.01; // optional stop limit
opt.surface_target = RefinementTarget::ridges; // omit unrelated valley sheets

SurfaceMesh result = extract_height_ridges(f, bounds, opt);
```

The refinement criterion mirrors `refineLongestEdge` in the notebook. For
each tet, it aligns the selected Hessian eigenvectors, tests whether
`gradient dot eigenvector` has mixed signs at its four vertices, and applies
the convex/concave classification to select ridges, valleys, or both. The
eligible tet with the longest edge is selected next. `MTetMesh::split_edge()`
splits the full incident-edge ring, so the mesh remains conforming; derivatives
are evaluated only for the midpoint vertex created by that split.

If you already have an MTet coarse grid, pass it by reference instead. It is
the same grid that is refined and surfaced—there is no copied vertex or tet
array:

```cpp
#include <mtet/grid.h>

mtet::MTetMesh coarse_grid = mtet::generate_tet_grid(
    {12, 12, 12}, {-1, -1, -1}, {1, 1, 1}, mtet::TET6);
SurfaceMesh result = extract_height_ridges(f, coarse_grid, opt);
```

Python exposes the same settings:

```python
refine = rs.LongestEdgeRefinementOptions()
refine.target = rs.RefinementTarget.ridges_and_valleys
refine.max_splits = 500
refine.minimum_edge_length = 0.01
options.longest_edge_refinement = refine
```

`result.vertices` are the dual vertices (one centroid for each active Kuhn
tetrahedron); `ridge_triangles` and `valley_triangles` index that array.

## Correspondence to the notebooks

- `height_ridge.nb`: implemented. At vertices, the code evaluates the
  eigensystem of `-H`, classifies strong ridges/valleys by `k_max + k_min`,
  finds sign changes of `gradient dot curvature-direction` on edges, then uses
  the notebook's dual-cell surfacing construction.
- `kuhn.nb`: implemented internally as the six-tetrahedra-per-cube Kuhn grid.
- `vipss.nb` and `vipss_open.nb`: not included as they are function-fitting
  systems (HRBF/VIPSS), rather than the requested surfacer. Their fitted field
  can be passed to this library through its scalar or derivative callbacks.

The algorithm follows the notebook's convention: the largest eigenvector of
`-H` defines a ridge and the smallest defines a valley. The optional short
root refinement re-evaluates analytic derivatives along each crossing edge.

MTet owns and validates the oriented TET6 tetrahedral grid. The surfacer builds
its compact edge-incidence table by iterating MTet's tetrahedra, then Eigen's
`SelfAdjointEigenSolver<Matrix3d>` computes the ordered eigensystem of the
negated Hessian.

For a watertight interior surface, avoid placing an expected ridge/valley
exactly on a grid-vertex plane. For example, use an odd resolution along the
normal direction or shift the bounds slightly. Zero-valued whole edges are
deliberately ignored because they have no unique dual-cell topology.

The `ridge_example` program deliberately starts from a coarse MTet grid,
enables 300 ridge-driven longest-edge splits, and writes two ASCII PLY files
in its working directory: `ridge_example.ply` is the extracted ridge surface;
`adaptive_grid_wireframe.ply` contains the final MTet vertex/edge wireframe.
