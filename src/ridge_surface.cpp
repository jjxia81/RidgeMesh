#include "ridge_surface/ridge_surface.hpp"

#include <Eigen/Eigenvalues>
#include <mtet/grid.h>
#include <mtet/mtet.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace ridge_surface {
namespace {

constexpr double kZeroTolerance = 1e-12;
constexpr std::size_t kNoSurfaceVertex = std::numeric_limits<std::size_t>::max();

struct OrderedEigenSystem {
    std::array<double, 3> values;
    std::array<Vec3, 3> vectors;
};

struct GridEdge {
    mtet::VertexId first_vertex;
    mtet::VertexId second_vertex;
    std::vector<std::size_t> incident_tets;
};

struct CrossingEdge {
    mtet::VertexId first_vertex;
    mtet::VertexId second_vertex;
    std::vector<std::size_t> incident_tets;
    int label; // +1 = ridge, -1 = valley.
};

// MTet is the only owner of the oriented TET6 grid and its coordinates.
struct TetGrid {
    mtet::MTetMesh mesh;
};

struct VertexSample {
    Vec3 gradient;
    OrderedEigenSystem eigensystem;
    bool is_convex_dominant;
};

OrderedEigenSystem compute_eigensystem(const Mat3& matrix) {
    Eigen::Matrix3d eigen_matrix;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            eigen_matrix(row, column) = matrix[row][column];
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(eigen_matrix);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Hessian eigendecomposition failed");
    }

    OrderedEigenSystem result;
    // Eigen returns ascending values. Reverse to follow k1 >= k2 >= k3.
    for (int output_index = 0; output_index < 3; ++output_index) {
        const int eigen_index = 2 - output_index;
        result.values[output_index] = solver.eigenvalues()[eigen_index];
        result.vectors[output_index] = normalized({
            solver.eigenvectors()(0, eigen_index),
            solver.eigenvectors()(1, eigen_index),
            solver.eigenvectors()(2, eigen_index),
        });
    }
    return result;
}

TetGrid make_kuhn_grid(const Bounds3D& bounds, int nx, int ny, int nz) {
    TetGrid grid;
    // MTet's TET6 is an oriented, six-tetrahedra-per-cell Kuhn grid.
    grid.mesh = mtet::generate_tet_grid(
        {static_cast<std::size_t>(nx), static_cast<std::size_t>(ny), static_cast<std::size_t>(nz)},
        {static_cast<float>(bounds.min.x), static_cast<float>(bounds.min.y), static_cast<float>(bounds.min.z)},
        {static_cast<float>(bounds.max.x), static_cast<float>(bounds.max.y), static_cast<float>(bounds.max.z)},
        mtet::TET6);

    return grid;
}

Vec3 position_of(const mtet::MTetMesh& mesh, mtet::VertexId vertex_id) {
    const auto position = mesh.get_vertex(vertex_id);
    return {position[0], position[1], position[2]};
}

// Build the compact edge -> incident-tetrahedra table needed for dual surfacing.
std::vector<GridEdge> build_edge_adjacency(const TetGrid& grid) {
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> edge_index_by_vertices;
    std::vector<GridEdge> edges;

    std::size_t tet_index = 0;
    grid.mesh.seq_foreach_tet([&](mtet::TetId, auto tet_ids) {
        for (int first = 0; first < 4; ++first) {
            for (int second = first + 1; second < 4; ++second) {
                const auto key = std::minmax(
                    tet_ids[first].value_of(),
                    tet_ids[second].value_of());
                const auto [iterator, inserted] = edge_index_by_vertices.emplace(key, edges.size());
                if (inserted) {
                    edges.push_back({tet_ids[first], tet_ids[second], {tet_index}});
                } else {
                    edges[iterator->second].incident_tets.push_back(tet_index);
                }
            }
        }
        ++tet_index;
    });
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
    const std::vector<Vec3>& surface_vertices,
    const std::vector<std::size_t>& surface_vertex_by_tet,
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

        // Sort the dual vertices around the crossing edge, then triangulate.
        const Vec3 edge_axis = normalized(
            position_of(grid_mesh, edge.second_vertex) - position_of(grid_mesh, edge.first_vertex));
        const Vec3 basis_u = normalized(
            std::abs(edge_axis.x) < 0.8 ? cross(edge_axis, {1, 0, 0})
                                        : cross(edge_axis, {0, 1, 0}));
        const Vec3 basis_v = cross(edge_axis, basis_u);

        Vec3 center;
        for (const std::size_t vertex_index : ring) {
            center += surface_vertices[vertex_index];
        }
        center = center / static_cast<double>(ring.size());

        std::sort(ring.begin(), ring.end(), [&](std::size_t left, std::size_t right) {
            const Vec3 left_offset = surface_vertices[left] - center;
            const Vec3 right_offset = surface_vertices[right] - center;
            const double left_angle = std::atan2(dot(left_offset, basis_v), dot(left_offset, basis_u));
            const double right_angle = std::atan2(dot(right_offset, basis_v), dot(right_offset, basis_u));
            return left_angle < right_angle;
        });

        for (std::size_t index = 1; index + 1 < ring.size(); ++index) {
            output_triangles.push_back({{ring[0], ring[index], ring[index + 1]}});
        }
    }
}

Vec3 axis_step(int axis, double step) {
    if (axis == 0) return {step, 0, 0};
    if (axis == 1) return {0, step, 0};
    return {0, 0, step};
}

} // namespace

SurfaceMesh extract_height_ridges(
    const DifferentialField3D& field,
    const Bounds3D& bounds,
    const SurfaceOptions& options) {
    if (!field.gradient || !field.hessian || options.nx < 1 || options.ny < 1 || options.nz < 1) {
        throw std::invalid_argument("valid field and positive grid resolution required");
    }

    const TetGrid tet_grid = make_kuhn_grid(bounds, options.nx, options.ny, options.nz);

    // 1. Evaluate derivatives and classify each grid vertex.
    std::unordered_map<std::uint64_t, VertexSample> samples_by_vertex_id;
    samples_by_vertex_id.reserve(tet_grid.mesh.get_num_vertices());
    tet_grid.mesh.seq_foreach_vertex([&](mtet::VertexId vertex_id, auto position) {
        const Vec3 point = {position[0], position[1], position[2]};

        Mat3 negated_hessian = field.hessian(point);
        for (auto& row : negated_hessian) {
            for (double& value : row) value = -value;
        }

        const OrderedEigenSystem eigensystem = compute_eigensystem(negated_hessian);
        samples_by_vertex_id.emplace(vertex_id.value_of(), VertexSample{
            field.gradient(point),
            eigensystem,
            eigensystem.values[0] + eigensystem.values[2] > 0.0,
        });
    });

    // 2. Locate eligible ridge and valley crossings on grid edges.
    const std::size_t tet_count = tet_grid.mesh.get_num_tets();
    std::vector<Vec3> crossing_sum_by_tet(tet_count);
    std::vector<int> crossing_count_by_tet(tet_count);
    std::vector<CrossingEdge> crossing_edges;

    for (const GridEdge& edge : build_edge_adjacency(tet_grid)) {
        const VertexSample& first_sample = samples_by_vertex_id.at(edge.first_vertex.value_of());
        const VertexSample& second_sample = samples_by_vertex_id.at(edge.second_vertex.value_of());
        const bool ridge = first_sample.is_convex_dominant && second_sample.is_convex_dominant;
        const bool valley = !first_sample.is_convex_dominant && !second_sample.is_convex_dominant;
        if (!ridge && !valley) continue;

        const int label = ridge ? 1 : -1;
        const int direction_index = ridge ? 0 : 2;
        Vec3 crossing_point;
        const bool found_crossing = options.subdivide_roots
            ? find_refined_crossing(field, position_of(tet_grid.mesh, edge.first_vertex),
                  position_of(tet_grid.mesh, edge.second_vertex), first_sample.gradient,
                  second_sample.gradient, first_sample.eigensystem.vectors[direction_index],
                  second_sample.eigensystem.vectors[direction_index], ridge, options, crossing_point)
            : find_linear_crossing(position_of(tet_grid.mesh, edge.first_vertex),
                  position_of(tet_grid.mesh, edge.second_vertex), first_sample.gradient,
                  second_sample.gradient, first_sample.eigensystem.vectors[direction_index],
                  second_sample.eigensystem.vectors[direction_index], crossing_point);
        if (!found_crossing) continue;

        crossing_edges.push_back({edge.first_vertex, edge.second_vertex, edge.incident_tets, label});
        for (const std::size_t tet_index : edge.incident_tets) {
            crossing_sum_by_tet[tet_index] += crossing_point;
            ++crossing_count_by_tet[tet_index];
        }
    }

    // 3. Create one dual vertex per active tetrahedron.
    SurfaceMesh surface;
    std::vector<std::size_t> surface_vertex_by_tet(tet_count, kNoSurfaceVertex);
    for (std::size_t tet_index = 0; tet_index < tet_count; ++tet_index) {
        if (crossing_count_by_tet[tet_index] == 0) continue;
        surface_vertex_by_tet[tet_index] = surface.vertices.size();
        surface.vertices.push_back(
            crossing_sum_by_tet[tet_index] / static_cast<double>(crossing_count_by_tet[tet_index]));
    }

    // 4. Make one dual polygon per crossing edge and split it into triangles.
    append_surface_polygons(crossing_edges, 1, tet_grid.mesh, surface.vertices,
        surface_vertex_by_tet, surface.ridge_triangles);
    append_surface_polygons(crossing_edges, -1, tet_grid.mesh, surface.vertices,
        surface_vertex_by_tet, surface.valley_triangles);
    return surface;
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
