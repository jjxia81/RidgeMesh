#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

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
};

struct Bounds3D { Vec3 min, max; };

enum class RefinementTarget {
  none,
  ridges,
  valleys,
  ridges_and_valleys,
};

// Adaptive, conforming longest-edge bisection of the initial TET6 grid.
// A zero minimum_edge_length matches the notebook: every tetrahedron that can
// contain the selected ridge/valley condition is eligible until max_splits.
struct LongestEdgeRefinementOptions {
  RefinementTarget target = RefinementTarget::none;
  int max_splits = 0;
  double minimum_edge_length = 0.0;
};

struct SurfaceOptions {
  int nx = 32, ny = 32, nz = 32;       // cells along x/y/z
  // The coarse TET6 grid is refined before surfacing when enabled.
  LongestEdgeRefinementOptions longest_edge_refinement;
  bool subdivide_roots = true;          // bracket-preserving root refinement
  int root_iterations = 4;
  double root_tolerance = 1e-7;
  double finite_difference_step = 1e-4; // only used by the scalar-field overload
};
struct Triangle { std::array<std::size_t,3> indices; };
struct SurfaceMesh {
  std::vector<Vec3> vertices;
  std::vector<Triangle> ridge_triangles;
  std::vector<Triangle> valley_triangles;
};

// Extract strong height ridges and valleys, following height_ridge.nb.
// Requires C++20 because the internal MTet connectivity dependency uses it.
// The output is a dual mesh: one vertex per active tetrahedron.
SurfaceMesh extract_height_ridges(const DifferentialField3D& field,
                                  const Bounds3D& bounds,
                                  const SurfaceOptions& options = {});

// Convenience overload. Derivatives are central finite differences; provide
// DifferentialField3D instead when analytic derivatives are available.
SurfaceMesh extract_height_ridges(const ScalarField3D& field,
                                  const Bounds3D& bounds,
                                  const SurfaceOptions& options = {});

} // namespace ridge_surface
