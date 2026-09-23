#include <cassert>
#include <cmath>
#include <stdexcept>
#include <mtet/grid.h>
#include "ridge_surface/ridge_surface.hpp"
int main() {
  using namespace ridge_surface;
  SurfaceOptions options{13, 12, 12};
  options.retain_dual_polygons = true;
  options.longest_edge_refinement.target = RefinementTarget::ridges;
  options.longest_edge_refinement.max_splits = 2;
  auto mesh=extract_height_ridges(ScalarField3D([](const Vec3& p) { return -p.x*p.x-.1*p.y*p.y-.05*p.z*p.z; }), {{-1,-1,-1},{1,1,1}}, options);
  assert(!mesh.vertices.empty());
  assert(!mesh.ridge_triangles.empty());
  assert(!mesh.ridge_polygons.empty());

  DifferentialField3D analytic_field{
    [](const Vec3& p) { return Vec3{-2.0 * p.x, -0.2 * p.y, -0.1 * p.z}; },
    [](const Vec3&) { return Mat3{{{{-2, 0, 0}}, {{0, -0.2, 0}}, {{0, 0, -0.1}}}}; },
  };
  mtet::MTetMesh coarse_grid = mtet::generate_tet_grid(
      {13, 12, 12}, {-1, -1, -1}, {1, 1, 1}, mtet::TET6);
  const std::size_t initial_vertex_count = coarse_grid.get_num_vertices();
  auto refined_mesh = extract_height_ridges(analytic_field, coarse_grid, options);
  assert(coarse_grid.get_num_vertices() > initial_vertex_count);
  assert(!refined_mesh.ridge_triangles.empty());
  std::size_t triangle_index = 0;
  for (const Polygon& polygon : refined_mesh.ridge_polygons) {
    if (polygon.indices.size() < 3 ||
        polygon.indices.size() > refined_mesh.ridge_triangles.size() - triangle_index) {
      throw std::runtime_error("invalid center-fan polygon");
    }
    const std::size_t center_index = refined_mesh.ridge_triangles[triangle_index].indices[0];
    if (center_index < refined_mesh.dual_vertex_count) {
      throw std::runtime_error("center fan did not add a polygon-center vertex");
    }
    Vec3 expected_center{};
    for (const std::size_t vertex_index : polygon.indices) {
      expected_center += refined_mesh.vertices[vertex_index];
    }
    expected_center = expected_center / static_cast<double>(polygon.indices.size());
    if (norm(refined_mesh.vertices[center_index] - expected_center) > 1e-10) {
      throw std::runtime_error("polygon-center vertex is not the ring mean");
    }
    for (std::size_t index = 0; index < polygon.indices.size(); ++index) {
      const auto& triangle = refined_mesh.ridge_triangles[triangle_index++].indices;
      if (triangle[0] != center_index || triangle[1] != polygon.indices[index] ||
          triangle[2] != polygon.indices[(index + 1) % polygon.indices.size()]) {
        throw std::runtime_error("center-fan triangle does not match polygon boundary");
      }
    }
  }
  if (triangle_index != refined_mesh.ridge_triangles.size()) {
    throw std::runtime_error("center-fan triangle count does not match polygons");
  }

  SurfaceOptions vertex_fan_options = options;
  vertex_fan_options.polygon_triangulation = PolygonTriangulation::vertex_fan;
  vertex_fan_options.longest_edge_refinement.max_splits = 0;
  mtet::MTetMesh vertex_fan_grid = mtet::generate_tet_grid(
      {13, 12, 12}, {-1, -1, -1}, {1, 1, 1}, mtet::TET6);
  const auto vertex_fan_mesh = extract_height_ridges(analytic_field, vertex_fan_grid, vertex_fan_options);
  std::size_t expected_triangle_count = 0;
  for (const Polygon& polygon : vertex_fan_mesh.ridge_polygons) {
    expected_triangle_count += polygon.indices.size() - 2;
  }
  if (vertex_fan_mesh.ridge_polygons.empty() ||
      vertex_fan_mesh.vertices.size() != vertex_fan_mesh.dual_vertex_count ||
      expected_triangle_count != vertex_fan_mesh.ridge_triangles.size()) {
    throw std::runtime_error("notebook-style vertex fan changed");
  }
}
