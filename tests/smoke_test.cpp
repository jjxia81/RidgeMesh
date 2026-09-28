#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <map>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <mtet/grid.h>
#include "ridge_surface/ridge_surface.hpp"
int main() try {
  using namespace ridge_surface;
  const LongestEdgeRefinementOptions default_refinement;
  if (default_refinement.refine_curvature_boundary || default_refinement.refine_orientation_boundary ||
      default_refinement.linearity_absolute_tolerance != 0.1) {
    throw std::runtime_error("boundary and linearity defaults are incorrect");
  }
  if (SurfaceOptions{}.ridge_curvature_filter != RidgeCurvatureFilter::crossing_point) {
    throw std::runtime_error("crossing-point curvature filter is not the default");
  }
  if (SurfaceOptions{}.check_crossing_point_curvature) {
    throw std::runtime_error("crossing-point curvature step 3 should default to disabled");
  }
  // One endpoint fails k1+k2, the other passes. The actual root has k1+k3=.2.
  const DifferentialField3D crossing_curvature_field{
    [](const Vec3& p) { return Vec3{-2 * p.x, 0, 0}; },
    [](const Vec3& p) {
      return Mat3{{{{-(1.7 + 1.8 * p.x), 0, 0}}, {{0, 1, 0}},
                   {{0, 0, -(-1.5 + 1.5 * p.x)}}}};
    },
  };
  SurfaceOptions curvature_options;
  curvature_options.nx = 1;
  curvature_options.ny = curvature_options.nz = 2;
  curvature_options.surface_target = RefinementTarget::ridges;
  curvature_options.ridge_curvature_filter = RidgeCurvatureFilter::both_endpoints;
  const Bounds3D curvature_bounds{{-1, -1, -1}, {1, 1, 1}};
  if (!extract_height_ridges(crossing_curvature_field, curvature_bounds, curvature_options).vertices.empty()) {
    throw std::runtime_error("original both-endpoint curvature filtering changed");
  }
  curvature_options.ridge_curvature_filter = RidgeCurvatureFilter::crossing_point;
  for (const bool refine_roots : {true, false}) {
    curvature_options.subdivide_roots = refine_roots;
    if (extract_height_ridges(crossing_curvature_field, curvature_bounds, curvature_options).ridge_triangles.empty()) {
      throw std::runtime_error("either-endpoint precheck lost a valid crossing-point ridge");
    }
    curvature_options.minimum_curvature_sum = 1.0;
    curvature_options.check_crossing_point_curvature = false;
    if (extract_height_ridges(crossing_curvature_field, curvature_bounds, curvature_options).ridge_triangles.empty()) {
      throw std::runtime_error("disabled step 3 still rejected crossing-point curvature");
    }
    curvature_options.check_crossing_point_curvature = true;
    if (!extract_height_ridges(crossing_curvature_field, curvature_bounds, curvature_options).vertices.empty()) {
      throw std::runtime_error("crossing-point curvature threshold was not enforced");
    }
    curvature_options.minimum_curvature_sum = 0.0;
  }
  const std::array<Vec3, 4> unit_tet{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  int gradient_queries = 0;
  int hessian_queries = 0;
  DifferentialField3D sample_test_field{
    [&](const Vec3& p) { ++gradient_queries; return Vec3{p.x * p.x, 0, 0}; },
    [&](const Vec3&) {
      ++hessian_queries;
      return Mat3{{{{-2, 0, 0}}, {{0, -.2, 0}}, {{0, 0, -.1}}}};
    },
  };
  for (const bool use_sixteen : {true, false}) {
    gradient_queries = hessian_queries = 0;
    const double error = use_sixteen ? directional_linearity_error_16(sample_test_field, unit_tet)
                                    : directional_linearity_error_5(sample_test_field, unit_tet);
    const int expected_queries = use_sixteen ? 20 : 9; // includes four vertices
    if (std::abs(error - 2.0 / 9.0) > 1e-12 || gradient_queries != expected_queries ||
        hessian_queries != expected_queries) {
      throw std::runtime_error("linearity error or new-sample count is incorrect");
    }
  }
  sample_test_field.gradient = [](const Vec3& p) { return Vec3{2 * p.x - p.y + 3 * p.z + 1, 0, 0}; };
  if (directional_linearity_error_16(sample_test_field, unit_tet) > 1e-12 ||
      directional_linearity_error_5(sample_test_field, unit_tet) > 1e-12) {
    throw std::runtime_error("affine directional condition is not recognized as linear");
  }
  // Synthetic derivative callbacks isolate refinement decisions independently
  // of surfacing. No directional values change sign in these fixtures.
  const DifferentialField3D curvature_boundary_field{
    [](const Vec3&) { return Vec3{1, 1, 1}; },
    [](const Vec3& p) {
      return Mat3{{{{-(p.x + 3), 0, 0}}, {{0, 0, 0}}, {{0, 0, -(p.x - 3)}}}};
    },
  };
  const DifferentialField3D inconsistent_directions_field{
    [](const Vec3&) { return Vec3{1, 0, 0}; },
    [](const Vec3& p) {
      const Vec3 direction = p.x < 0 ? Vec3{1, 0, 0}
          : Vec3{.5, p.y < 0 ? std::sqrt(.75) : -std::sqrt(.75), 0};
      const std::array<double, 3> components{direction.x, direction.y, direction.z};
      Mat3 hessian{};
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          hessian[row][column] = -2 * components[row] * components[column]
                              - (row == column ? 1.0 : 0.0);
        }
      }
      return hessian;
    },
  };
  const auto verify_split_count = [&](const DifferentialField3D& field,
                                     RefinementPipeline pipeline, int budget,
                                     double minimum_length, std::size_t expected_splits,
                                     bool refine_boundary = false,
                                     LinearitySampling sampling = LinearitySampling::disabled,
                                     double absolute_tolerance = 1e-3,
                                     bool refine_orientation = true) {
    mtet::MTetMesh grid;
    const auto v0 = grid.add_vertex(-1, 0, 0);
    const auto v1 = grid.add_vertex(1, -1, 0);
    const auto v2 = grid.add_vertex(1, 1, 0);
    const auto v3 = grid.add_vertex(-1, 0, 1);
    grid.add_tet(v0, v1, v2, v3);
    const auto initial_count = grid.get_num_vertices();
    SurfaceOptions test_options;
    test_options.surface_target = RefinementTarget::ridges;
    test_options.longest_edge_refinement.target = RefinementTarget::ridges;
    test_options.longest_edge_refinement.pipeline = pipeline;
    test_options.longest_edge_refinement.max_splits = budget;
    test_options.longest_edge_refinement.minimum_edge_length = minimum_length;
    test_options.longest_edge_refinement.refine_curvature_boundary = refine_boundary;
    test_options.longest_edge_refinement.refine_orientation_boundary = refine_orientation;
    test_options.longest_edge_refinement.linearity_sampling = sampling;
    test_options.longest_edge_refinement.linearity_absolute_tolerance = absolute_tolerance;
    extract_height_ridges(field, grid, test_options);
    if (grid.get_num_vertices() != initial_count + expected_splits) {
      throw std::runtime_error("adaptive pipeline decision or stopping limit is incorrect: pipeline="
          + std::to_string(static_cast<int>(pipeline)) + ", budget=" + std::to_string(budget)
          + ", expected=" + std::to_string(expected_splits) + ", actual="
          + std::to_string(grid.get_num_vertices() - initial_count));
    }
  };
  if (LongestEdgeRefinementOptions{}.pipeline != RefinementPipeline::curvature_orientation) {
    throw std::runtime_error("new adaptive pipeline is not the default");
  }
  verify_split_count(curvature_boundary_field, RefinementPipeline::curvature_orientation, 1, 0, 0);
  verify_split_count(curvature_boundary_field, RefinementPipeline::curvature_orientation, 1, 0, 1, true);
  verify_split_count(curvature_boundary_field, RefinementPipeline::curvature_orientation, 10, 4, 0, true);
  DifferentialField3D boundary_crossing_field = curvature_boundary_field;
  boundary_crossing_field.gradient = [](const Vec3& p) { return Vec3{-p.x, 0, 0}; };
  verify_split_count(boundary_crossing_field, RefinementPipeline::curvature_orientation, 1, 0, 1);
  verify_split_count(inconsistent_directions_field, RefinementPipeline::curvature_orientation, 1, 0, 1);
  // Vertex-0 alignment shows no crossing here, but independent pair alignment
  // detects the crossing on the inconsistent pair. Disabling Boundary I must
  // still consider that edge instead of silently dropping the tet.
  verify_split_count(inconsistent_directions_field, RefinementPipeline::curvature_orientation,
                     1, 0, 1, false, LinearitySampling::disabled, 1e-3, false);
  DifferentialField3D zero_condition_field = inconsistent_directions_field;
  zero_condition_field.gradient = [](const Vec3&) { return Vec3{}; };
  verify_split_count(zero_condition_field, RefinementPipeline::curvature_orientation,
                     1, 0, 1, false, LinearitySampling::disabled, 1e-3, true);
  verify_split_count(zero_condition_field, RefinementPipeline::curvature_orientation,
                     1, 0, 0, false, LinearitySampling::disabled, 1e-3, false);
  verify_split_count(inconsistent_directions_field, RefinementPipeline::curvature_orientation,
                     1, 4, 0, false, LinearitySampling::disabled, 1e-3, false);
  for (const auto& field : {curvature_boundary_field, inconsistent_directions_field}) {
    verify_split_count(field, RefinementPipeline::legacy_crossing, 1, 0, 0);
    verify_split_count(field, RefinementPipeline::curvature_orientation, 0, 0, 0);
    verify_split_count(field, RefinementPipeline::curvature_orientation, 10, 4, 0);
    verify_split_count(field, RefinementPipeline::legacy_crossing, 10, 4, 0);
  }
  SurfaceOptions options{13, 12, 12};
  // Existing crossing-driven regressions retain the pre-linearity behavior.
  options.longest_edge_refinement.linearity_sampling = LinearitySampling::disabled;
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
  for (const auto sampling : {LinearitySampling::cubic_16, LinearitySampling::face_centers_5}) {
    // Linearity is the final gate: nonlinearity alone is not eligibility.
    verify_split_count(analytic_field, RefinementPipeline::curvature_orientation, 1, 0, 0, false, sampling);
    DifferentialField3D nonlinear_field = analytic_field;
    gradient_queries = 0;
    nonlinear_field.gradient = [&](const Vec3& p) { ++gradient_queries; return Vec3{p.x * p.x, 0, 0}; };
    verify_split_count(nonlinear_field, RefinementPipeline::curvature_orientation, 1, 0, 0, false, sampling);
    if (gradient_queries != 4) {
      throw std::runtime_error("non-crossing tet made additional linearity queries");
    }
    nonlinear_field.gradient = [](const Vec3& p) { return Vec3{p.x * p.x * p.x, 0, 0}; };
    verify_split_count(nonlinear_field, RefinementPipeline::curvature_orientation, 1, 0, 1, false, sampling);
    verify_split_count(nonlinear_field, RefinementPipeline::curvature_orientation, 1, 0, 0, false, sampling, 10);
    verify_split_count(nonlinear_field, RefinementPipeline::curvature_orientation, 1, 4, 0, false, sampling);
    nonlinear_field.hessian = [](const Vec3&) {
      return Mat3{{{{.1, 0, 0}}, {{0, .2, 0}}, {{0, 0, 2}}}};
    };
    verify_split_count(nonlinear_field, RefinementPipeline::curvature_orientation, 1, 0, 0, false, sampling);
  }
  for (const auto pipeline : {RefinementPipeline::curvature_orientation,
                             RefinementPipeline::legacy_crossing}) {
    verify_split_count(analytic_field, pipeline, 1, 0, 1);
    verify_split_count(analytic_field, pipeline, 0, 0, 0);
    verify_split_count(analytic_field, pipeline, 10, 4, 0);
  }
  mtet::MTetMesh coarse_grid = mtet::generate_tet_grid(
      {13, 12, 12}, {-1, -1, -1}, {1, 1, 1}, mtet::TET6);
  const std::size_t initial_vertex_count = coarse_grid.get_num_vertices();
  auto refined_mesh = extract_height_ridges(analytic_field, coarse_grid, options);
  assert(coarse_grid.get_num_vertices() > initial_vertex_count);
  assert(!refined_mesh.ridge_triangles.empty());

  SurfaceOptions many_splits = options;
  many_splits.longest_edge_refinement.max_splits = 100;
  mtet::MTetMesh queue_grid = mtet::generate_tet_grid(
      {4, 4, 4}, {-1.05, -1, -1}, {0.95, 1, 1}, mtet::TET6);
  const std::size_t queue_initial_vertices = queue_grid.get_num_vertices();
  const auto queue_surface = extract_height_ridges(analytic_field, queue_grid, many_splits);
  if (queue_grid.get_num_vertices() != queue_initial_vertices + 100 ||
      queue_surface.ridge_triangles.empty()) {
    throw std::runtime_error("candidate queue lost a split or the ridge surface");
  }
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

  SurfaceOptions polygon_options = vertex_fan_options;
  polygon_options.polygon_triangulation = PolygonTriangulation::polygons_only;
  polygon_options.retain_dual_polygons = false; // polygon mode must retain faces itself
  mtet::MTetMesh polygon_grid = mtet::generate_tet_grid(
      {13, 12, 12}, {-1, -1, -1}, {1, 1, 1}, mtet::TET6);
  const auto polygon_mesh = extract_height_ridges(analytic_field, polygon_grid, polygon_options);
  if (polygon_mesh.ridge_polygons.empty() || !polygon_mesh.ridge_triangles.empty() ||
      polygon_mesh.vertices.size() != polygon_mesh.dual_vertex_count) {
    throw std::runtime_error("polygon-only ridge mode returned triangles or no polygons");
  }
  for (const Polygon& polygon : polygon_mesh.ridge_polygons) {
    if (polygon.indices.size() < 3) {
      throw std::runtime_error("polygon-only ridge face has fewer than three vertices");
    }
    for (const std::size_t index : polygon.indices) {
      if (index >= polygon_mesh.dual_vertex_count) {
        throw std::runtime_error("polygon-only ridge face references a non-dual vertex");
      }
    }
  }

  const DifferentialField3D valley_field{
    [](const Vec3& p) { return Vec3{2.0 * p.x, 0.2 * p.y, 0.1 * p.z}; },
    [](const Vec3&) { return Mat3{{{{2, 0, 0}}, {{0, 0.2, 0}}, {{0, 0, 0.1}}}}; },
  };
  polygon_options.surface_target = RefinementTarget::valleys;
  mtet::MTetMesh valley_grid = mtet::generate_tet_grid(
      {13, 12, 12}, {-1, -1, -1}, {1, 1, 1}, mtet::TET6);
  const auto valley_mesh = extract_height_ridges(valley_field, valley_grid, polygon_options);
  if (valley_mesh.valley_polygons.empty() || !valley_mesh.valley_triangles.empty() ||
      valley_mesh.vertices.size() != valley_mesh.dual_vertex_count) {
    throw std::runtime_error("polygon-only valley mode returned triangles or no polygons");
  }

  // A flattened ellipsoid on an irregular adaptive grid exposed polygon
  // angle-sorting defects. Every edge of this interior surface must be shared.
  constexpr std::array<double, 3> inverse_radii_squared{1.0 / .5625, 1.0 / .5625, 16.0};
  const auto ellipsoid_offset = [&](const Vec3& point) {
    return point.x * point.x * inverse_radii_squared[0]
         + point.y * point.y * inverse_radii_squared[1]
         + point.z * point.z * inverse_radii_squared[2] - 1.0;
  };
  const DifferentialField3D ellipsoid_field{
    [&](const Vec3& point) {
      const double scale = -4.0 * ellipsoid_offset(point);
      return Vec3{scale * inverse_radii_squared[0] * point.x,
                  scale * inverse_radii_squared[1] * point.y,
                  scale * inverse_radii_squared[2] * point.z};
    },
    [&](const Vec3& point) {
      const std::array<double, 3> coordinates{point.x, point.y, point.z};
      Mat3 hessian{};
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          hessian[row][column] = -8.0 * inverse_radii_squared[row] * coordinates[row]
              * inverse_radii_squared[column] * coordinates[column];
          if (row == column) {
            hessian[row][column] -= 4.0 * ellipsoid_offset(point) * inverse_radii_squared[row];
          }
        }
      }
      return hessian;
    },
  };
  SurfaceOptions ellipsoid_options;
  ellipsoid_options.surface_target = RefinementTarget::ridges;
  ellipsoid_options.polygon_triangulation = PolygonTriangulation::polygons_only;
  ellipsoid_options.longest_edge_refinement.target = RefinementTarget::ridges;
  // Preserve the original irregular-grid fixture for the ring-order regression.
  ellipsoid_options.longest_edge_refinement.pipeline = RefinementPipeline::legacy_crossing;
  ellipsoid_options.longest_edge_refinement.max_splits = 100000;
  ellipsoid_options.longest_edge_refinement.minimum_edge_length = .005;
  auto ellipsoid_grid = mtet::generate_tet_grid(
      {4, 4, 4}, {-.98, -1, -.40}, {1.02, 1, .40}, mtet::TET6);
  const auto ellipsoid_mesh = extract_height_ridges(ellipsoid_field, ellipsoid_grid, ellipsoid_options);
  if (ellipsoid_mesh.ridge_polygons.empty()) {
    throw std::runtime_error("adaptive ellipsoid returned no polygons");
  }
  std::map<std::pair<std::size_t, std::size_t>, int> polygon_edge_counts;
  for (const Polygon& polygon : ellipsoid_mesh.ridge_polygons) {
    for (std::size_t index = 0; index < polygon.indices.size(); ++index) {
      const auto first = polygon.indices[index];
      const auto second = polygon.indices[(index + 1) % polygon.indices.size()];
      ++polygon_edge_counts[{std::min(first, second), std::max(first, second)}];
    }
  }
  for (const auto& [edge, count] : polygon_edge_counts) {
    if (count != 2) {
      throw std::runtime_error("adaptive ellipsoid polygon connectivity is not watertight");
    }
  }
  // Exercise the default boundary/orientation pipeline on the real field too.
  ellipsoid_options.longest_edge_refinement.pipeline = RefinementPipeline::curvature_orientation;
  ellipsoid_options.longest_edge_refinement.max_splits = 1000;
  auto default_ellipsoid_grid = mtet::generate_tet_grid(
      {4, 4, 4}, {-.98, -1, -.40}, {1.02, 1, .40}, mtet::TET6);
  const auto initial_default_vertices = default_ellipsoid_grid.get_num_vertices();
  const auto default_ellipsoid_mesh = extract_height_ridges(
      ellipsoid_field, default_ellipsoid_grid, ellipsoid_options);
  if (default_ellipsoid_mesh.ridge_polygons.empty() ||
      default_ellipsoid_grid.get_num_vertices() <= initial_default_vertices ||
      default_ellipsoid_grid.get_num_vertices() > initial_default_vertices + 1000) {
    throw std::runtime_error("default adaptive ellipsoid failed or exceeded its split budget");
  }
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
