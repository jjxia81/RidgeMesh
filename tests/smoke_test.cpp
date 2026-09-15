#include <cassert>
#include <mtet/grid.h>
#include "ridge_surface/ridge_surface.hpp"
int main() {
  using namespace ridge_surface;
  SurfaceOptions options{13, 12, 12};
  options.longest_edge_refinement.target = RefinementTarget::ridges;
  options.longest_edge_refinement.max_splits = 2;
  auto mesh=extract_height_ridges(ScalarField3D([](const Vec3& p) { return -p.x*p.x-.1*p.y*p.y-.05*p.z*p.z; }), {{-1,-1,-1},{1,1,1}}, options);
  assert(!mesh.vertices.empty());
  assert(!mesh.ridge_triangles.empty());

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
}
