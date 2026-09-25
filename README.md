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

Install RidgeMesh as a Python package (including its native extension):

```bash
python -m pip install ".[torch,plot]"
```

The `torch` extra supplies the GeoUDF/PyTorch adapter; `plot` supplies
Matplotlib. The package can then be imported as `ridgemesh`. The default CMake
build also produces a native module named `ridge_surface` in the build output
directory for direct local use. On Windows, after a Release build:

```powershell
$env:PYTHONPATH = "$(Resolve-Path .\vs2022\Release)"
python -c "import ridge_surface; print(ridge_surface.__doc__)"
```

```python
import ridge_surface as rs
import ridge_surface_plot as rsp

radius = 0.6
bounds = rs.Bounds3D(rs.Vec3(-1.05, -1, -1), rs.Vec3(.95, 1, 1))

# Exact derivatives of f(x) = -(||x||^2 - radius^2)^2.
def gradient(x, y, z):
    offset = x*x + y*y + z*z - radius*radius
    return (-4*offset*x, -4*offset*y, -4*offset*z)

def hessian(x, y, z):
    point = (x, y, z)
    offset = x*x + y*y + z*z - radius*radius
    return [[-8*point[i]*point[j] - (4*offset if i == j else 0)
             for j in range(3)] for i in range(3)]

uniform = rs.SurfaceOptions()
uniform.nx = uniform.ny = uniform.nz = 64
uniform.surface_target = rs.RefinementTarget.ridges
uniform_mesh = rs.extract_height_ridges_from_derivatives(gradient, hessian, bounds, uniform)

adaptive = rs.SurfaceOptions()
adaptive.nx = adaptive.ny = adaptive.nz = 4
adaptive.surface_target = rs.RefinementTarget.ridges
adaptive.longest_edge_refinement.target = rs.RefinementTarget.ridges
adaptive.longest_edge_refinement.max_splits = 5000
adaptive.longest_edge_refinement.minimum_edge_length = .005
adaptive_mesh = rs.extract_height_ridges_from_derivatives(gradient, hessian, bounds, adaptive)

figure, axes = rsp.plot_mesh_comparison(uniform_mesh, adaptive_mesh)
figure.savefig("sphere_mesh_comparison.png", dpi=180)

figure, axes = rsp.plot_radial_error(
    {"uniform 64^3": uniform_mesh, "adaptive": adaptive_mesh}, radius)
figure.savefig("sphere_radial_error.png", dpi=180)
```

For accuracy and speed, use `extract_height_ridges_from_derivatives` with
Python callbacks returning a length-3 gradient and a 3-by-3 Hessian.

### PyTorch / GeoUDF

`extract_torch_udf` accepts a model following GeoUDF's
`model(input_dict, query)` convention. It evaluates the scalar field and
computes its gradient and Hessian with the selected derivative methods.
Set `gradient_mode="network"` to use a matching gradient returned as the
model's second output. For GeoUDF's field `F=-UDF²`, return
`(F, -2 * UDF * udf_grad)` from the wrapper; `udf_grad` is a learned,
unit-normalized direction, so this is an approximation rather than the exact
derivative of `F`.

```python
import ridgemesh as rm

options = rm.SurfaceOptions()
options.nx = options.ny = options.nz = 12
options.surface_target = rm.RefinementTarget.ridges
options.longest_edge_refinement.target = rm.RefinementTarget.ridges
options.longest_edge_refinement.max_splits = 1000
options.longest_edge_refinement.minimum_edge_length = .01

# `model` is a loaded GeoUDF torch model and `input_dict` is its prepared
# context. The adapter constructs GeoUDF queries with shape (1, 3, M).
def ridge_field(context, query):
    udf, udf_grad = model(context, query)
    return -(udf * udf), -2 * udf.unsqueeze(-1) * udf_grad

mesh = rm.extract_torch_udf(
    ridge_field,
    input_dict,
    rm.Bounds3D(rm.Vec3(-1, -1, -1), rm.Vec3(1, 1, 1)),
    options,
    device="cuda",
    derivative_mode="finite_difference",
    gradient_mode="finite_difference",
    finite_difference_step=5e-3,
)

from ridgemesh import plot
figure, axes = plot.plot_mesh(mesh, show_valleys=False, title="GeoUDF ridge")
figure.savefig("geoudf_ridge.png", dpi=180)
```

The library adapter defaults to autograd for both derivatives. The GeoUDF batch
script defaults to central function-value differences for both the gradient
and Hessian. For the same combination through the library API, set
`gradient_mode="finite_difference"` and `derivative_mode="finite_difference"`.
The 3-D Hessian stencil needs 19 scalar-field samples per grid vertex; uniform
grid samples are batched when `uniform_grid_batch_size` is positive.

To use GeoUDF's learned gradient instead, select `gradient_mode="network"`.
It can be paired with the same numerical Hessian, or with
`hessian_backend="gradient_difference"` to form a Hessian from differences of
the learned gradient. The gradient and Hessian sources are independent.

To test that finite-difference path independently of a trained model, run the
known-sphere smoke test. It uses a smooth scalar field with its ridge exactly
at radius `0.6`, verifies the resulting radial error, and writes
`numerical_hessian_sphere.ply`:

```bash
python examples/python_numerical_hessian_sphere.py
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

The first `result.dual_vertex_count` entries of `result.vertices` are the
tetrahedron dual vertices (averages of crossing points). With the default
center-fan triangulation, the remaining entries are polygon centers;
`ridge_triangles` and `valley_triangles` index the full vertex array.

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

The `ridge_example` program compares a uniform `64 x 64 x 64` TET6 grid with
an adaptive `4 x 4 x 4` MTet grid using up to 5,000 longest-edge splits and a
0.005 minimum edge length. It writes three ASCII PLY files in its
working directory: `uniform_64_sphere.ply` is the dense uniform-grid ridge;
`ridge_example.ply` is the adaptive-grid ridge; and
`adaptive_grid_wireframe.ply` contains the final adaptive MTet vertex/edge
wireframe.

`ellipsoid_example` is a separate analytic example for
\(F(x) = -(x^2/a^2 + y^2/b^2 + z^2/c^2 - 1)^2\). It writes
`ellipsoid_ridge.ply`, a uniform-grid ridge mesh for a flattened sphere
with semi-axes \(a=b=0.75\) and \(c=0.25\). It uses the default center-fan
triangulation and writes only this one mesh. Set
`SurfaceOptions::retain_dual_polygons` to access the ordered dual-vertex rings
from the C++ API; the option is disabled by default to avoid storing them for
large neural fields.

The example also accepts `--triangulation vertex_fan` or
`--triangulation polygons_only`. Each invocation still writes just one PLY:
`ellipsoid_ridge_vertex_fan.ply` or `ellipsoid_ridge_polygons.ply`, respectively.

`SurfaceOptions::polygon_triangulation` selects one of three face outputs:

- `PolygonTriangulation::center_fan` (default): add the arithmetic mean of
  each polygon's dual vertices and connect it to every boundary edge.
- `PolygonTriangulation::vertex_fan`: triangulate from the polygon's first
  dual vertex, as in the Mathematica `tess` function; no centers are added.
- `PolygonTriangulation::polygons_only`: return ordered polygon faces in
  `ridge_polygons` / `valley_polygons`, with no triangles or center vertices.
  This mode retains the polygons even when `retain_dual_polygons` is false.

For example, set `options.polygon_triangulation =
PolygonTriangulation::polygons_only` in C++, or
`options.polygon_triangulation = rs.PolygonTriangulation.polygons_only` in
Python. `SurfaceMesh::dual_vertex_count` separates the original tetrahedron
dual vertices from any appended centers; in the latter two modes, it equals
`vertices.size()`.
