#include "adaptive_refine.hpp"
#include "ridge_surface/ridge_surface.hpp"

#include <mtet/grid.h>
#include <mtet/mtet.h>
#include <nanothread/nanothread.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ridge_surface {
namespace {

using detail::OrderedEigenSystem;
using detail::VertexSample;
using detail::compute_eigensystem;
using detail::evaluate_vertex_sample;
using detail::position_of;
using detail::refine_longest_edges_in_place;

constexpr double kZeroTolerance = 1e-12;
constexpr std::size_t kNoSurfaceVertex = std::numeric_limits<std::size_t>::max();

struct GridEdge {
    mtet::VertexId first_vertex;
    mtet::VertexId second_vertex;
    std::vector<std::size_t> incident_tets;
    mtet::EdgeId mtet_edge;
};

struct CrossingEdge {
    mtet::VertexId first_vertex;
    mtet::VertexId second_vertex;
    std::vector<std::size_t> incident_tets;
    int label; // +1 = ridge, -1 = valley.
};

struct EdgeCrossingResult {
    Vec3 point;
    int label = 0; // Zero means the edge produced no accepted crossing.
};

mtet::MTetMesh make_kuhn_grid(const Bounds3D& bounds, int nx, int ny, int nz) {
    // MTet's TET6 is an oriented, six-tetrahedra-per-cell Kuhn grid.
    return mtet::generate_tet_grid(
        {static_cast<std::size_t>(nx), static_cast<std::size_t>(ny), static_cast<std::size_t>(nz)},
        {static_cast<float>(bounds.min.x), static_cast<float>(bounds.min.y), static_cast<float>(bounds.min.z)},
        {static_cast<float>(bounds.max.x), static_cast<float>(bounds.max.y), static_cast<float>(bounds.max.z)},
        mtet::TET6);
}

// Build the compact edge -> incident-tetrahedra table needed for dual surfacing.
std::vector<GridEdge> build_edge_adjacency(mtet::MTetMesh& mesh) {
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> edge_index_by_vertices;
    std::unordered_map<std::uint64_t, std::size_t> tet_index_by_id;
    std::vector<GridEdge> edges;

    std::size_t tet_index = 0;
    mesh.seq_foreach_tet([&](mtet::TetId tet_id, auto) {
        tet_index_by_id.emplace(tet_id.value_of(), tet_index++);
        mesh.foreach_edge_in_tet(tet_id, [&](mtet::EdgeId edge_id, mtet::VertexId first,
                                              mtet::VertexId second) {
            const std::uint64_t first_id = first.value_of();
            const std::uint64_t second_id = second.value_of();
            const std::pair<std::uint64_t, std::uint64_t> key{
                std::min(first_id, second_id), std::max(first_id, second_id)};
            const auto [iterator, inserted] = edge_index_by_vertices.emplace(key, edges.size());
            if (inserted) {
                edges.push_back({first, second, {}, edge_id});
            }
        });
    });

    // Preserve the topological one-ring order. Geometric angle sorting can
    // change which neighboring tetrahedra share a polygon boundary edge.
    for (GridEdge& edge : edges) {
        mesh.foreach_tet_around_edge(edge.mtet_edge, [&](mtet::TetId tet_id) {
            const mtet::TetId canonical = mesh.get_edge_tet(mesh.get_edge(tet_id, 0));
            edge.incident_tets.push_back(tet_index_by_id.at(canonical.value_of()));
        });
    }
    return edges;
}

Vec3 interpolate_root(Vec3 first_point, Vec3 second_point, double first_value, double second_value) {
    const double denominator = second_value - first_value;
    if (std::abs(denominator) < kZeroTolerance) {
        return (first_point + second_point) * 0.5;
    }
    return (first_point * second_value - second_point * first_value) / denominator;
}

bool find_linear_crossing(
    Vec3 first_point,
    Vec3 second_point,
    Vec3 first_gradient,
    Vec3 second_gradient,
    Vec3 first_direction,
    Vec3 second_direction,
    Vec3& crossing_point) {
    double first_value = dot(first_gradient, first_direction);
    double second_value = dot(second_gradient, second_direction);

    // Eigenvector signs are arbitrary. Align directions before comparing values.
    if (dot(first_direction, second_direction) < 0.0) {
        second_value = -second_value;
    }

    // A whole zero edge has no unique dual-cell topology.
    if (first_value * second_value >= 0.0) {
        return false;
    }

    crossing_point = interpolate_root(first_point, second_point, first_value, second_value);
    return true;
}

Vec3 curvature_direction_at(
    const DifferentialField3D& field,
    Vec3 point,
    Vec3 reference_direction,
    bool ridge) {
    Mat3 negated_hessian = field.hessian(point);
    for (auto& row : negated_hessian) {
        for (double& value : row) {
            value = -value;
        }
    }

    const OrderedEigenSystem eigen = compute_eigensystem(negated_hessian);
    Vec3 direction = eigen.vectors[ridge ? 0 : 2];
    if (dot(direction, reference_direction) < 0.0) {
        direction = direction * -1.0;
    }
    return direction;
}

bool find_refined_crossing(
    const DifferentialField3D& field,
    Vec3 first_point,
    Vec3 second_point,
    Vec3 first_gradient,
    Vec3 second_gradient,
    Vec3 first_direction,
    Vec3 second_direction,
    bool ridge,
    const SurfaceOptions& options,
    Vec3& crossing_point) {
    if (dot(first_direction, second_direction) < 0.0) {
        second_direction = second_direction * -1.0;
    }

    double first_value = dot(first_gradient, first_direction);
    double second_value = dot(second_gradient, second_direction);
    if (first_value * second_value >= 0.0) {
        return false;
    }

    for (int iteration = 0; iteration < options.root_iterations; ++iteration) {
        crossing_point = interpolate_root(first_point, second_point, first_value, second_value);
        const Vec3 direction = curvature_direction_at(field, crossing_point, first_direction, ridge);
        const double crossing_value = dot(field.gradient(crossing_point), direction);
        if (std::abs(crossing_value) < options.root_tolerance) {
            return true;
        }

        if (first_value * crossing_value <= 0.0) {
            second_point = crossing_point;
            second_value = crossing_value;
            second_direction = direction;
        } else {
            first_point = crossing_point;
            first_value = crossing_value;
            first_direction = direction;
        }
    }

    crossing_point = interpolate_root(first_point, second_point, first_value, second_value);
    return true;
}

void append_surface_polygons(
    const std::vector<CrossingEdge>& crossing_edges,
    int requested_label,
    const mtet::MTetMesh& grid_mesh,
    std::vector<Vec3>& surface_vertices,
    const std::vector<std::size_t>& surface_vertex_by_tet,
    PolygonTriangulation triangulation,
    std::vector<Polygon>* output_polygons,
    std::vector<Triangle>& output_triangles) {
    for (const CrossingEdge& edge : crossing_edges) {
        if (edge.label != requested_label) {
            continue;
        }

        std::vector<std::size_t> ring;
        for (const std::size_t tet_index : edge.incident_tets) {
            const std::size_t surface_vertex = surface_vertex_by_tet[tet_index];
            if (surface_vertex != kNoSurfaceVertex) {
                ring.push_back(surface_vertex);
            }
        }
        if (ring.size() < 3) {
            continue;
        }

        // MTet supplies the cyclic order. Only reverse the whole loop to set
        // its winding; never reorder individual vertices by their positions.
        const Vec3 edge_axis = normalized(
            position_of(grid_mesh, edge.second_vertex) - position_of(grid_mesh, edge.first_vertex));

        Vec3 center;
        for (const std::size_t vertex_index : ring) {
            center += surface_vertices[vertex_index];
        }
        center = center / static_cast<double>(ring.size());

        Vec3 area_vector{};
        for (std::size_t index = 0; index < ring.size(); ++index) {
            area_vector += cross(surface_vertices[ring[index]] - center,
                surface_vertices[ring[(index + 1) % ring.size()]] - center);
        }
        if (dot(area_vector, edge_axis) < 0.0) {
            std::reverse(ring.begin(), ring.end());
        }

        if (output_polygons != nullptr) {
            output_polygons->push_back({ring});
        }

        if (triangulation == PolygonTriangulation::polygons_only) {
            continue;
        }
        if (triangulation == PolygonTriangulation::center_fan) {
            const std::size_t center_index = surface_vertices.size();
            surface_vertices.push_back(center);
            for (std::size_t index = 0; index < ring.size(); ++index) {
                output_triangles.push_back({{
                    center_index, ring[index], ring[(index + 1) % ring.size()]}});
            }
        } else {
            for (std::size_t index = 1; index + 1 < ring.size(); ++index) {
                output_triangles.push_back({{ring[0], ring[index], ring[index + 1]}});
            }
        }
    }
}

Vec3 axis_step(int axis, double step) {
    if (axis == 0) return {step, 0, 0};
    if (axis == 1) return {0, step, 0};
    return {0, 0, step};
}

} // namespace

SurfaceMesh extract_height_ridges_from_grid(
    const DifferentialField3D& field,
    mtet::MTetMesh& coarse_grid,
    const SurfaceOptions& options) {
    if (!field.gradient || !field.hessian || coarse_grid.get_num_vertices() == 0 ||
        coarse_grid.get_num_tets() == 0 ||
        options.longest_edge_refinement.max_splits < 0 ||
        options.longest_edge_refinement.minimum_edge_length < 0.0 ||
        !std::isfinite(options.longest_edge_refinement.minimum_edge_length) ||
        options.longest_edge_refinement.curvature_zero_tolerance < 0.0 ||
        !std::isfinite(options.longest_edge_refinement.curvature_zero_tolerance) ||
        options.longest_edge_refinement.linearity_absolute_tolerance < 0.0 ||
        !std::isfinite(options.longest_edge_refinement.linearity_absolute_tolerance) ||
        options.longest_edge_refinement.linearity_relative_tolerance < 0.0 ||
        !std::isfinite(options.longest_edge_refinement.linearity_relative_tolerance)) {
        throw std::invalid_argument("valid field and non-empty MTet grid required");
    }
    if (std::isfinite(options.minimum_ridge_field_value) && !field.value) {
        throw std::invalid_argument("a scalar field callback is required for minimum_ridge_field_value");
    }

    // 1. Evaluate derivatives and classify each grid vertex.
    std::unordered_map<std::uint64_t, VertexSample> samples_by_vertex_id;
    samples_by_vertex_id.reserve(coarse_grid.get_num_vertices());
    if (options.parallel_initial_sampling) {
        // First pass: collect stable IDs and positions without concurrent MTet access.
        struct GridVertex {
            std::uint64_t id;
            Vec3 position;
        };
        std::vector<GridVertex> vertices;
        vertices.reserve(coarse_grid.get_num_vertices());
        coarse_grid.seq_foreach_vertex([&](mtet::VertexId id, auto position) {
            vertices.push_back({id.value_of(), {position[0], position[1], position[2]}});
        });

        // Second pass: each worker owns one output slot; no hash-map writes.
        std::vector<VertexSample> evaluated(vertices.size());
        std::exception_ptr evaluation_error;
        std::mutex error_mutex;
        drjit::parallel_for(drjit::blocked_range<std::size_t>(0, vertices.size(), 1),
            [&](drjit::blocked_range<std::size_t> range) {
                for (std::size_t index = range.begin(); index != range.end(); ++index) {
                    try {
                        evaluated[index] = evaluate_vertex_sample(field, vertices[index].position);
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!evaluation_error) evaluation_error = std::current_exception();
                    }
                }
            });
        if (evaluation_error) std::rethrow_exception(evaluation_error);

        // Publish samples only after all workers have finished.
        for (std::size_t index = 0; index < vertices.size(); ++index) {
            samples_by_vertex_id.emplace(vertices[index].id, evaluated[index]);
        }
    } else {
        coarse_grid.seq_foreach_vertex([&](mtet::VertexId vertex_id, auto position) {
            samples_by_vertex_id.emplace(vertex_id.value_of(),
                evaluate_vertex_sample(field, {position[0], position[1], position[2]}));
        });
    }

    // 2. Adapt the coarse MTet grid before locating surface crossings.
    // MTet splits the entire edge one-ring, preserving a conforming tet mesh.
    refine_longest_edges_in_place(
        coarse_grid, field, options.longest_edge_refinement, samples_by_vertex_id,
        options.minimum_curvature_sum);

    // 3. Locate eligible ridge and valley crossings on grid edges.
    const std::size_t tet_count = coarse_grid.get_num_tets();
    std::vector<Vec3> crossing_sum_by_tet(tet_count);
    std::vector<int> crossing_count_by_tet(tet_count);
    std::vector<CrossingEdge> crossing_edges;
    const std::vector<GridEdge> grid_edges = build_edge_adjacency(coarse_grid);
    const auto& vertex_samples = samples_by_vertex_id;

    // This phase reads the finished grid and derivative cache; each edge is independent.
    const auto evaluate_edge = [&](const GridEdge& edge) {
        const VertexSample& first_sample = vertex_samples.at(edge.first_vertex.value_of());
        const VertexSample& second_sample = vertex_samples.at(edge.second_vertex.value_of());
        const bool extract_ridges = options.surface_target == RefinementTarget::ridges ||
            options.surface_target == RefinementTarget::ridges_and_valleys;
        const bool extract_valleys = options.surface_target == RefinementTarget::valleys ||
            options.surface_target == RefinementTarget::ridges_and_valleys;
        const bool check_ridge_at_crossing = options.ridge_curvature_filter == RidgeCurvatureFilter::crossing_point;
        const auto endpoint_passes_precheck = [](const VertexSample& sample) {
            return sample.eigensystem.values[0] + sample.eigensystem.values[1] > 0.0;
        };
        const bool ridge = extract_ridges && (check_ridge_at_crossing
            ? endpoint_passes_precheck(first_sample) || endpoint_passes_precheck(second_sample)
            : first_sample.curvature_sum > options.minimum_curvature_sum &&
              second_sample.curvature_sum > options.minimum_curvature_sum);
        const bool valley = extract_valleys &&
            first_sample.curvature_sum < -options.minimum_curvature_sum &&
            second_sample.curvature_sum < -options.minimum_curvature_sum;

        EdgeCrossingResult result;
        if (!ridge && !valley) return result;

        const Vec3 first_point = position_of(coarse_grid, edge.first_vertex);
        const Vec3 second_point = position_of(coarse_grid, edge.second_vertex);
        const auto try_crossing = [&](bool is_ridge) {
            const int direction_index = is_ridge ? 0 : 2;
            Vec3 crossing_point;
            const bool found_crossing = options.subdivide_roots
                ? find_refined_crossing(field, first_point, second_point, first_sample.gradient,
                      second_sample.gradient, first_sample.eigensystem.vectors[direction_index],
                      second_sample.eigensystem.vectors[direction_index], is_ridge, options, crossing_point)
                : find_linear_crossing(first_point, second_point, first_sample.gradient,
                      second_sample.gradient, first_sample.eigensystem.vectors[direction_index],
                      second_sample.eigensystem.vectors[direction_index], crossing_point);
            if (!found_crossing) return false;
            if (is_ridge && check_ridge_at_crossing && options.check_crossing_point_curvature) {
                Mat3 negated_hessian = field.hessian(crossing_point);
                for (auto& row : negated_hessian) {
                    for (double& value : row) value = -value;
                }
                const OrderedEigenSystem eigen = compute_eigensystem(negated_hessian);
                if (!(eigen.values[0] + eigen.values[2] > options.minimum_curvature_sum)) {
                    return false;
                }
            }
            if (is_ridge && std::isfinite(options.minimum_ridge_field_value) &&
                field.value(crossing_point) < options.minimum_ridge_field_value) {
                return false;
            }
            result = {crossing_point, is_ridge ? 1 : -1};
            return true;
        };
        if (ridge) {
            const bool accepted = try_crossing(true);
            // A failed ridge-root test must not hide a valid valley crossing.
            if (!accepted && check_ridge_at_crossing && valley) try_crossing(false);
        } else if (valley) {
            try_crossing(false);
        }
        return result;
    };

    const auto record_crossing = [&](const GridEdge& edge, const EdgeCrossingResult& result) {
        if (result.label == 0) return;
        crossing_edges.push_back({edge.first_vertex, edge.second_vertex, edge.incident_tets, result.label});
        for (const std::size_t tet_index : edge.incident_tets) {
            crossing_sum_by_tet[tet_index] += result.point;
            ++crossing_count_by_tet[tet_index];
        }
    };

    if (options.parallel_edge_crossings) {
        std::vector<EdgeCrossingResult> results(grid_edges.size());
        std::exception_ptr evaluation_error;
        std::mutex error_mutex;
        drjit::parallel_for(drjit::blocked_range<std::size_t>(0, grid_edges.size(), 64),
            [&](drjit::blocked_range<std::size_t> range) {
                for (std::size_t index = range.begin(); index != range.end(); ++index) {
                    try {
                        results[index] = evaluate_edge(grid_edges[index]);
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!evaluation_error) evaluation_error = std::current_exception();
                    }
                }
            });
        if (evaluation_error) std::rethrow_exception(evaluation_error);
        // Reduce in grid-edge order to keep dual-vertex coordinates and face order stable.
        for (std::size_t index = 0; index < grid_edges.size(); ++index) {
            record_crossing(grid_edges[index], results[index]);
        }
    } else {
        for (const GridEdge& edge : grid_edges) {
            record_crossing(edge, evaluate_edge(edge));
        }
    }

    // 4. Create one dual vertex per active tetrahedron.
    SurfaceMesh surface;
    std::vector<std::size_t> surface_vertex_by_tet(tet_count, kNoSurfaceVertex);
    for (std::size_t tet_index = 0; tet_index < tet_count; ++tet_index) {
        if (crossing_count_by_tet[tet_index] == 0) continue;
        surface_vertex_by_tet[tet_index] = surface.vertices.size();
        surface.vertices.push_back(
            crossing_sum_by_tet[tet_index] / static_cast<double>(crossing_count_by_tet[tet_index]));
    }

    surface.dual_vertex_count = surface.vertices.size();

    // 5. Make one dual polygon per crossing edge; triangulation is optional.
    const bool retain_polygons = options.retain_dual_polygons ||
        options.polygon_triangulation == PolygonTriangulation::polygons_only;
    append_surface_polygons(crossing_edges, 1, coarse_grid, surface.vertices,
        surface_vertex_by_tet,
        options.polygon_triangulation,
        retain_polygons ? &surface.ridge_polygons : nullptr,
        surface.ridge_triangles);
    append_surface_polygons(crossing_edges, -1, coarse_grid, surface.vertices,
        surface_vertex_by_tet,
        options.polygon_triangulation,
        retain_polygons ? &surface.valley_polygons : nullptr,
        surface.valley_triangles);
    return surface;
}

SurfaceMesh extract_height_ridges(
    const DifferentialField3D& field,
    const Bounds3D& bounds,
    const SurfaceOptions& options) {
    if (options.nx < 1 || options.ny < 1 || options.nz < 1) {
        throw std::invalid_argument("positive coarse-grid resolution required");
    }
    mtet::MTetMesh coarse_grid = make_kuhn_grid(bounds, options.nx, options.ny, options.nz);
    return extract_height_ridges_from_grid(field, coarse_grid, options);
}

SurfaceMesh extract_height_ridges(
    const DifferentialField3D& field,
    mtet::MTetMesh& coarse_grid,
    const SurfaceOptions& options) {
    return extract_height_ridges_from_grid(field, coarse_grid, options);
}

SurfaceMesh extract_height_ridges(
    const ScalarField3D& scalar_field,
    const Bounds3D& bounds,
    const SurfaceOptions& options) {
    if (!scalar_field) {
        throw std::invalid_argument("field callback is required");
    }

    const double step = options.finite_difference_step;
    const auto evaluate = [&](Vec3 point) { return scalar_field(point); };

    DifferentialField3D field;
    field.value = evaluate;
    field.gradient = [=](Vec3 point) {
        const double dx = (evaluate(point + axis_step(0, step)) - evaluate(point - axis_step(0, step))) / (2.0 * step);
        const double dy = (evaluate(point + axis_step(1, step)) - evaluate(point - axis_step(1, step))) / (2.0 * step);
        const double dz = (evaluate(point + axis_step(2, step)) - evaluate(point - axis_step(2, step))) / (2.0 * step);
        return Vec3{dx, dy, dz};
    };

    field.hessian = [=](Vec3 point) {
        Mat3 hessian{};
        const double center_value = evaluate(point);
        for (int row = 0; row < 3; ++row) {
            const Vec3 row_step = axis_step(row, step);
            hessian[row][row] =
                (evaluate(point + row_step) - 2.0 * center_value + evaluate(point - row_step)) / (step * step);
            for (int column = row + 1; column < 3; ++column) {
                const Vec3 column_step = axis_step(column, step);
                const double mixed =
                    (evaluate(point + row_step + column_step) - evaluate(point + row_step - column_step) -
                     evaluate(point - row_step + column_step) + evaluate(point - row_step - column_step)) /
                    (4.0 * step * step);
                hessian[row][column] = mixed;
                hessian[column][row] = mixed;
            }
        }
        return hessian;
    };

    return extract_height_ridges(field, bounds, options);
}

} // namespace ridge_surface
