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
model's second output. For the optional GeoUDF network-gradient mode, use
`F=-UDF` and return `(F, -udf_grad)` from the wrapper. GeoUDF's `udf_grad`
is a learned, unit-normalized direction, so it is not guaranteed to equal
the exact derivative of `UDF`.

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
    udf, _ = model(context, query)
    return -(udf * udf)

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

The GeoUDF batch option `--ridgemesh-numerical-negative-udf` applies that same
numerical gradient/Hessian stencil to `-UDF` instead of the default `-UDF²`.

To use GeoUDF's learned gradient instead, wrap the model to return
`(-udf, -udf_grad)` and select `gradient_mode="network"`. The numerical Hessian
then comes from central differences of that negated network gradient, without
the 19-point scalar-value stencil or a `-2 * udf` scaling factor. Each uniform
grid batch queries the center and six axis offsets in one model call; adaptive
points use the same seven-point gradient stencil. This option approximates
ridges of `-UDF`, which is less smooth at zero than the default `-UDF²` field.

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

For thread-safe native C++ gradient and Hessian callbacks, initial-grid
sampling is parallel by default (`opt.parallel_initial_sampling = true`). It
uses two passes: collect vertex IDs and positions, evaluate into independent
slots in parallel, then populate the ID-indexed cache serially. Set the option
to `false` for callbacks that are not thread-safe. This parallelizes only initial-grid derivative
sampling, not adaptive splitting or polygon assembly. Python exposes the same
option, but Python callbacks acquire the GIL and therefore do not run
concurrently; use `uniform_grid_batch_size` in `extract_torch_udf` for batched
neural-field queries instead.

Directional zero crossings are evaluated on separate grid edges concurrently
by default (`opt.parallel_edge_crossings = true`). Set it to `false` for
non-thread-safe callbacks or to compare with the serial path. The edge-to-tet
accumulation remains serial and ordered, preserving the same dual mesh.
This option also requires thread-safe gradient, Hessian, and optional scalar
value callbacks. Python exposes `parallel_edge_crossings`, but Python callbacks
remain GIL-serialized.

The ellipsoid command-line examples enable both parallel stages by default.
Use `--no-parallel-initial-sampling` or `--no-parallel-edge-crossings` to
disable either stage. The corresponding positive flags remain accepted.

The previous `RefinementPipeline::curvature_orientation` follows the ridge-only
flowchart with a sampled directional-condition linearity check. For each tet:

1. Compute `s = k1 + k3` from the eigenvalues of `-H` at all four vertices.
   Boundary II is disabled by default (`refine_curvature_boundary = false`).
   Mixed signs or values within `curvature_zero_tolerance` of zero continue
   to the orientation/crossing checks, rather than automatically refining.
   Set `refine_curvature_boundary = true` to restore that automatic trigger.
2. Apply ridge/valley curvature eligibility. All negative vertex curvature sums
   reject a ridge tet (valley mode mirrors this). Mixed-curvature tets continue.
3. By default, align endpoint eigenvectors independently on each of the six
   edges and test for directional crossings. If Boundary I is enabled, instead
   align directions to vertex 0 and check remaining pairwise dot products;
   a negative pair then triggers orientation-boundary refinement even without
   a crossing.
4. Require an oriented `c = gradient dot eigenvector` crossing and sufficient
   curvature strength. A tet failing this gate makes no linearity sample queries.
5. As the final eligibility test, compare re-evaluated c at additional barycentric
   locations with `sum_i weight_i*c(vertex_i)`. Refine if the maximum absolute
   difference exceeds `linearity_absolute_tolerance + linearity_relative_tolerance
   * max_abs_c` (scale includes the vertex and additional sampled values).
   The default is `LinearitySampling::cubic_16`: twelve edge samples at 1/3 and
   2/3 plus four face centers, with absolute tolerance `0.1` and relative
   tolerance `0`. This mode has no tet-center sample. The cheaper alternative,
   `LinearitySampling::face_centers_5`, uses four face centers and the tet center.
   Eigenvectors at all sample locations are aligned to the vertex-0 reference.
   No derivative of c or Bezier-coefficient construction is required.
   Sufficiently linear crossings stop refining; nonlinearity alone does not
   trigger subdivision. Boundary I (and optional Boundary II) remain independent
   refinement triggers before this final test.

Set `linearity_sampling = LinearitySampling::disabled` to restore refinement
   based on directional crossings and curvature strength. In that mode, all
   negative vertex curvature sums reject a ridge tet (valley mode mirrors this).
   For mixed-curvature tets, at least one vertex must exceed the ridge threshold
   (or its sign-mirrored valley threshold); final extraction still checks both
   endpoints of each emitted edge crossing. Uniform-sign tets require all four.
   With sampling enabled, sufficiently linear crossing tets stop refining;
   nonlinear tets without an observed vertex crossing are not refined. Final
   mesh extraction still applies the curvature and crossing conditions.

Python selection:

```python
refine.linearity_sampling = rs.LinearitySampling.cubic_16  # default
refine.linearity_sampling = rs.LinearitySampling.face_centers_5  # cheaper alternative
refine.linearity_absolute_tolerance = 0.1
refine.linearity_relative_tolerance = 0.0
```

### Sampled Bezier-simplex refinement pipeline

Select `RefinementPipeline::bezier_simplex` in C++,
`rs.RefinementPipeline.bezier_simplex` in Python, or
`--refinement-pipeline bezier_simplex` in the ellipsoid examples. This is an
default adaptive pipeline; `curvature_orientation` remains available explicitly.

For each tet, this pipeline follows these checks in order:

1. Evaluate derivatives/eigenpairs at the four cached grid vertices and sixteen
   additional cubic barycentric locations (twelve edge locations and four face
   centers). Continue only if any sampled `k1+k3 > 0`.
2. Require a strict directional crossing on at least one of the six original
   tet edges, with endpoint eigenvectors aligned independently per edge. The
   additional samples do not add grid edges or change this crossing test.
3. If the twenty curvature sums straddle zero (including the configured
   `curvature_zero_tolerance` uncertainty band), refine.
4. Otherwise check tet-wide eigenvector orientability; refine if inconsistent.
5. Otherwise refine only if the maximum sampled directional-condition
   interpolation error exceeds the linearity tolerance. Reuse the sixteen
   evaluated samples; `face_centers_5` instead reuses their four face centers
   and evaluates one additional tet center. `disabled` bypasses this final test.

Boundary I/II checks are intrinsic to this pipeline regardless of the older
`refine_orientation_boundary` / `refine_curvature_boundary` flags. Ridge steps
use v1; valley targets mirror curvature signs and use v3. The positive-curvature
gate is a zero-sign test, independent of `minimum_curvature_sum`; surfacing
settings are unchanged. These are actual field samples at Bezier domain
locations, not fitted Bezier coefficients, and there is no interior sample in
the sixteen-location gate. An original edge whose endpoints show no crossing
is not subdivided into probe segments by this test, so small hidden surfaces
can still be missed. Both geometric and split-budget limits remain enforced.

```powershell
.\vs2022\Release\ellipsoid_linearity_example.exe --refinement-pipeline bezier_simplex --le 0.1
```

For `curvature_orientation`, Boundary I and Boundary II flags are both disabled by default. The default
`refine.refine_orientation_boundary = False` tests each of the
six tet edges independently. The second endpoint's eigenvector is sign-aligned
to the first before testing for a strict c sign change. Any crossing edge makes
the tet eligible for the final linearity test; curvature and stopping limits
still apply. The `legacy_crossing` pipeline is unchanged and ignores this option.
Set `refine.refine_orientation_boundary = True` to enable Boundary I, and
`refine.refine_curvature_boundary = True` to enable Boundary II (use `true` in C++).
The ellipsoid CLI provides `--enable-boundary-i` and `--enable-boundary-ii`;
the corresponding `--disable-boundary-i` / `--disable-boundary-ii` flags are also accepted.
Edge-local crossings can disagree around a non-orientable tet; this option does
not guarantee consistent topology or prevent all off-surface refinement. With
linearity enabled, its existing tet-reference interpolation remains a heuristic
on those tets. Use `--linearity-sampling disabled` to test edge crossings alone.

The separate `ellipsoid_linearity_example` starts from an adaptive 4x4x4 grid
and lets you control the absolute tolerance from the command line:

```powershell
.\vs2022\Release\ellipsoid_linearity_example.exe --le 0.01
```

On Ubuntu build the same CMake target and run
`./build/ellipsoid_linearity_example --le 0.01`.
The original `--linearity-absolute-tolerance` spelling remains accepted as an alias.
The default filter during **surfacing only** is
`--ridge-curvature-filter both_endpoints` (the flag may be omitted). Before
testing an oriented directional crossing, **both** edge endpoints must satisfy
`k1+k3 > max(0, minimum_curvature_sum)`, using eigenvalues of `-H`. The default
threshold is zero, and a negative strength setting cannot admit a nonpositive
ridge endpoint. This changes neither adaptive eligibility nor valley
filtering and does not guarantee watertightness; on a coarse grid, the strict
endpoint condition can miss a ridge between the endpoints.

The optional `--ridge-curvature-filter crossing_point` instead requires
`k1+k2 > 0` at either endpoint. With
`--check-crossing-point-curvature`, it also evaluates the Hessian at the root
and requires `k1+k3 > minimum_curvature_sum` there. Without that flag, the
root-curvature check is disabled. The crossing-point example mesh gets a
`_crossing_point` suffix for comparison. In Python set
`options.ridge_curvature_filter = rs.RidgeCurvatureFilter.crossing_point`;
in C++ use `RidgeCurvatureFilter::crossing_point`.
Use `--linearity-sampling face_centers_5` for the cheaper test, `--max-splits`
and `--minimum-edge-length` for the stopping limits, and `--help` for all options.
The ellipsoid examples default to 10000000 splits, a 0.005 edge cutoff, and
`cubic_16` sampling; large runs may consume substantial memory and disk space.
relative tolerance is zero. This example writes `ellipsoid_linearity_ridge.ply`
and `ellipsoid_linearity_grid_wireframe.ply` in the current directory, separate
from the original example's output names. Each rerun replaces these files.

The public C++/Python functions `directional_linearity_error_16` and
`directional_linearity_error_5` return the maximum difference for a supplied tet.

Set `opt.longest_edge_refinement.pipeline = RefinementPipeline::legacy_crossing`
to retain the previous curvature-classification plus crossing criterion.
In Python use `refine.pipeline = rs.RefinementPipeline.legacy_crossing`;
`rs.RefinementPipeline.curvature_orientation` selects the previous pipeline;
`rs.RefinementPipeline.bezier_simplex` selects the default sampled pipeline.
These choices do not enable adaptation by themselves: set `target` and
`max_splits` as above. Uniform extraction is unchanged.

All pipelines obey `max_splits`, counting actual MTet edge-split operations
(one split can replace multiple incident tets), and `minimum_edge_length`:
a candidate is discarded when its **longest edge** is at or below this length.
This is a refinement cutoff, not a guarantee that every child edge has that
minimum length. Coarse vertex-only tests can still miss an interior crossing;
refinement does not guarantee a closed surface when the budget is exhausted.

The eligible tet with the longest edge is selected next. Candidates are kept in a
descending-length queue indexed by tetrahedron ID. After a split, candidates
for the replaced edge-ring tetrahedra are removed, and only eligible new
tetrahedra are inserted; the whole grid is not rescanned. `MTetMesh::split_edge()`
splits the full incident-edge ring, so the mesh remains conforming; derivatives
are cached at grid vertices; sampled linearity additionally re-evaluates
derivatives at the 16 or 5 test locations for each checked tet. These probes
are not inserted as grid vertices. The legacy pipeline ignores linearity settings.

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
an adaptive `4 x 4 x 4` MTet grid using up to 100,000 longest-edge splits and a
0.005 minimum selected-edge length. This high split cap can take substantial
time and memory. It writes three ASCII PLY files in its
working directory: `uniform_64_sphere.ply` is the dense uniform-grid ridge;
`ridge_example.ply` is the adaptive-grid ridge; and
`adaptive_grid_wireframe.ply` contains the final adaptive MTet vertex/edge
wireframe.

`ellipsoid_example` is a separate analytic example for
\(F(x) = -(x^2/a^2 + y^2/b^2 + z^2/c^2 - 1)^2\). It writes
`ellipsoid_ridge.ply`, a uniform 64-cells-per-axis ridge mesh for a flattened sphere
with semi-axes \(a=b=0.75\) and \(c=0.25\). It uses the default center-fan
triangulation. Run `ellipsoid_example --res 128` for a uniform 128-cells-per-axis
grid; this writes `ellipsoid_ridge_res128.ply` and requires
substantially more time and memory than the default. `--res` is for uniform
runs only. Run `ellipsoid_example --adaptive` to start instead from a
`4 x 4 x 4` MTet grid and apply up to 10,000,000 longest-edge splits around the
ridge. That run writes `ellipsoid_adaptive_ridge.ply` and
`ellipsoid_adaptive_grid_wireframe.ply`; the latter shows the refined grid edges.
Set
`SurfaceOptions::retain_dual_polygons` to access the ordered dual-vertex rings
from the C++ API; the option is disabled by default to avoid storing them for
large neural fields.

The example also accepts `--triangulation vertex_fan` or
`--triangulation polygons_only`, with or without `--adaptive`. Each invocation
writes one ridge-mesh PLY, adding `_vertex_fan` or `_polygons` to its filename
for these alternate modes. The adaptive run additionally writes its grid
wireframe PLY.

`SurfaceOptions::polygon_triangulation` selects one of three face outputs:

Polygon boundaries follow MTet's cyclic tetrahedron adjacency around each
crossing edge, not geometric angle sorting of the dual vertices. This preserves
shared polygon edges on irregular adaptive grids. The arithmetic-mean center
is used only for center-fan triangulation; it does not determine connectivity.

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
