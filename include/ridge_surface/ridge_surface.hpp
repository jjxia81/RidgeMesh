#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mtet/mtet.h>

namespace ridge_surface {

struct Vec3 {
  double x{}, y{}, z{};
  Vec3& operator+=(const Vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
  Vec3& operator-=(const Vec3& b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
};
inline Vec3 operator+(Vec3 a, const Vec3& b) { return a += b; }
inline Vec3 operator-(Vec3 a, const Vec3& b) { return a -= b; }
inline Vec3 operator*(Vec3 a, double s) { return {a.x*s,a.y*s,a.z*s}; }
inline Vec3 operator*(double s, Vec3 a) { return a*s; }
inline Vec3 operator/(Vec3 a, double s) { return a*(1.0/s); }
inline double dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
inline double norm(Vec3 a) { return std::sqrt(dot(a,a)); }
inline Vec3 normalized(Vec3 a) { const double n=norm(a); return n>0 ? a/n : Vec3{}; }

using Mat3 = std::array<std::array<double,3>,3>;
using ScalarField3D = std::function<double(const Vec3&)>;
struct DifferentialField3D {
  std::function<Vec3(const Vec3&)> gradient;
  std::function<Mat3(const Vec3&)> hessian;
  std::function<double(const Vec3&)> value;
};

struct Bounds3D { Vec3 min, max; };

enum class RefinementTarget {
  none,
  ridges,
  valleys,
  ridges_and_valleys,
};

enum class PolygonTriangulation {
  center_fan, // add the polygon centroid and connect it to every boundary edge
  vertex_fan, // notebook's tess[poly]: fan from the first boundary vertex
  polygons_only, // keep dual polygons without making triangle or center vertices
};

enum class RefinementPipeline {
  curvature_orientation, // curvature boundary, tet orientability, then crossing
  legacy_crossing,       // previous curvature classification + crossing test
  bezier_simplex,        // 20-point curvature gate, edge crossing, boundaries, linearity
};

enum class LinearitySampling {
  cubic_16,       // twelve 1/3 and 2/3 edge locations + four face centers
  face_centers_5, // four face centers + tetrahedron center
  disabled,      // restore crossing-driven refinement without a linearity test
};

enum class RidgeCurvatureFilter {
  both_endpoints, // require k1+k3 > threshold at both endpoints (original)
  crossing_point, // either endpoint k1+k2 > 0; optional curvature check at root
};

// Adaptive, conforming longest-edge bisection of the initial TET6 grid.
// Stop when the candidate's longest edge is <= minimum_edge_length or when
// max_splits edge bisections have occurred. Zero disables the length cutoff.
struct LongestEdgeRefinementOptions {
  RefinementTarget target = RefinementTarget::none;
  int max_splits = 0;
  double minimum_edge_length = 0.0;
  RefinementPipeline pipeline = RefinementPipeline::bezier_simplex;
  double curvature_zero_tolerance = 1e-12; // new pipeline's Boundary II uncertainty band
  bool refine_curvature_boundary = false; // optional Boundary II trigger; new pipeline only
  bool refine_orientation_boundary = false; // false: skip Boundary I and use edge-local crossings
  LinearitySampling linearity_sampling = LinearitySampling::cubic_16;
  double linearity_absolute_tolerance = 0.1;
  double linearity_relative_tolerance = 0.0;
};

// Maximum |c(sample) - sum_i weight_i*c(vertex_i)|, where c = gradient dot
// the consistently oriented eigenvector of -H. direction_index 0 selects k1,
// 2 selects k3. These functions re-evaluate derivatives, not gradients of c.
double directional_linearity_error_16(const DifferentialField3D& field,
                                     const std::array<Vec3, 4>& tet_vertices,
                                     int direction_index = 0);
double directional_linearity_error_5(const DifferentialField3D& field,
                                    const std::array<Vec3, 4>& tet_vertices,
                                    int direction_index = 0);

struct SurfaceOptions {
  int nx = 32, ny = 32, nz = 32;       // cells along x/y/z
  // The coarse TET6 grid is refined before surfacing when enabled.
  LongestEdgeRefinementOptions longest_edge_refinement;
  // Choose which extracted surfaces are emitted. The default preserves the
  // original ridge-and-valley behavior.
  RefinementTarget surface_target = RefinementTarget::ridges_and_valleys;
  bool subdivide_roots = true;          // bracket-preserving root refinement
  int root_iterations = 4;
  double root_tolerance = 1e-7;
  // For a ridge, k1 + k3 is computed from the eigenvalues of -H. Require it
  // to exceed this value at the crossing point by default, or at both edge
  // endpoints with the original filter. Zero is a sign-only threshold.
  double minimum_curvature_sum = 0.0;
  RidgeCurvatureFilter ridge_curvature_filter = RidgeCurvatureFilter::crossing_point;
  bool check_crossing_point_curvature = false; // optional step 3; crossing_point mode only
  // If finite, a ridge crossing must have f(x) >= this value. This removes
  // local but low-valued ridges when extracting a particular level surface.
  // The default disables the gate.
  double minimum_ridge_field_value = -std::numeric_limits<double>::infinity();
  double finite_difference_step = 1e-4; // only used by the scalar-field overload
  bool retain_dual_polygons = false;     // optional with fan modes; implied by polygons_only
  PolygonTriangulation polygon_triangulation = PolygonTriangulation::center_fan;
};
struct Triangle { std::array<std::size_t,3> indices; };
struct Polygon { std::vector<std::size_t> indices; };
struct SurfaceMesh {
  std::vector<Vec3> vertices;
  // The first dual_vertex_count entries are tetrahedron dual vertices.
  // Center-fan triangulation appends one more vertex per polygon after them.
  std::size_t dual_vertex_count = 0;
  std::vector<Triangle> ridge_triangles;
  std::vector<Triangle> valley_triangles;
  std::vector<Polygon> ridge_polygons;
  std::vector<Polygon> valley_polygons;
};

// Extract strong height ridges and valleys, following height_ridge.nb.
// Requires C++20 because the internal MTet connectivity dependency uses it.
// The output is a dual mesh: one vertex per active tetrahedron.
SurfaceMesh extract_height_ridges(const DifferentialField3D& field,
                                  const Bounds3D& bounds,
                                  const SurfaceOptions& options = {});

// Surface an existing coarse MTet grid. When longest_edge_refinement is
// enabled, this mesh is refined in place; MTet remains its sole owner.
SurfaceMesh extract_height_ridges(const DifferentialField3D& field,
                                  mtet::MTetMesh& coarse_grid,
                                  const SurfaceOptions& options = {});

// Convenience overload. Derivatives are central finite differences; provide
// DifferentialField3D instead when analytic derivatives are available.
SurfaceMesh extract_height_ridges(const ScalarField3D& field,
                                  const Bounds3D& bounds,
                                  const SurfaceOptions& options = {});

} // namespace ridge_surface
