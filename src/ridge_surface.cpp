#include "ridge_surface/ridge_surface.hpp"

#include <Eigen/Eigenvalues>
#include <mtet/grid.h>
#include <mtet/mtet.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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
    mtet::EdgeId mtet_edge;
};

struct CrossingEdge {
    mtet::VertexId first_vertex;
    mtet::VertexId second_vertex;
    std::vector<std::size_t> incident_tets;
    int label; // +1 = ridge, -1 = valley.
};

struct VertexSample {
    Vec3 gradient;
    OrderedEigenSystem eigensystem;
    double curvature_sum;
};

struct RefinementCandidate {
    mtet::TetId tet_id;
    std::uint8_t longest_local_edge;
    double longest_edge_length;
};

struct LongestCandidateFirst {
    bool operator()(const RefinementCandidate& left, const RefinementCandidate& right) const {
        if (left.longest_edge_length != right.longest_edge_length) {
            return left.longest_edge_length > right.longest_edge_length;
        }
        return left.tet_id.value_of() < right.tet_id.value_of();
    }
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

mtet::MTetMesh make_kuhn_grid(const Bounds3D& bounds, int nx, int ny, int nz) {
    // MTet's TET6 is an oriented, six-tetrahedra-per-cell Kuhn grid.
    return mtet::generate_tet_grid(
        {static_cast<std::size_t>(nx), static_cast<std::size_t>(ny), static_cast<std::size_t>(nz)},
        {static_cast<float>(bounds.min.x), static_cast<float>(bounds.min.y), static_cast<float>(bounds.min.z)},
        {static_cast<float>(bounds.max.x), static_cast<float>(bounds.max.y), static_cast<float>(bounds.max.z)},
        mtet::TET6);
}

Vec3 position_of(const mtet::MTetMesh& mesh, mtet::VertexId vertex_id) {
    const auto position = mesh.get_vertex(vertex_id);
    return {position[0], position[1], position[2]};
}

VertexSample evaluate_vertex_sample(const DifferentialField3D& field, Vec3 point) {
    Mat3 negated_hessian = field.hessian(point);
    for (auto& row : negated_hessian) {
        for (double& value : row) {
            value = -value;
        }
    }

    const OrderedEigenSystem eigensystem = compute_eigensystem(negated_hessian);
    return {
        field.gradient(point),
        eigensystem,
        eigensystem.values[0] + eigensystem.values[2],
    };
}

void add_vertex_sample(
    const mtet::MTetMesh& mesh,
    const DifferentialField3D& field,
    mtet::VertexId vertex_id,
    std::unordered_map<std::uint64_t, VertexSample>& samples_by_vertex_id) {
    samples_by_vertex_id.try_emplace(
        vertex_id.value_of(), evaluate_vertex_sample(field, position_of(mesh, vertex_id)));
}

int sign_consistency(const std::array<double, 4>& values) {
    const auto sign_of = [](double value) {
        return value > 0.0 ? 1 : (value < 0.0 ? -1 : 0);
    };
    const int common_sign = sign_of(values[0]);
    for (std::size_t index = 1; index < values.size(); ++index) {
        const double value = values[index];
        if (common_sign != sign_of(value)) {
            return 0;
        }
    }
    return common_sign;
}

bool has_condition_crossing(
    std::span<const mtet::VertexId, 4> tet_vertices,
    const std::unordered_map<std::uint64_t, VertexSample>& samples_by_vertex_id,
    int direction_index) {
    const Vec3 reference_direction =
        samples_by_vertex_id.at(tet_vertices[0].value_of()).eigensystem.vectors[direction_index];
    std::array<double, 4> values;
    for (std::size_t index = 0; index < tet_vertices.size(); ++index) {
        const VertexSample& sample = samples_by_vertex_id.at(tet_vertices[index].value_of());
        Vec3 direction = sample.eigensystem.vectors[direction_index];
        if (dot(direction, reference_direction) < 0.0) {
            direction = direction * -1.0;
        }
        values[index] = dot(sample.gradient, direction);
    }
    return sign_consistency(values) == 0;
}

bool should_refine_tet_legacy(
    std::span<const mtet::VertexId, 4> tet_vertices,
    const std::unordered_map<std::uint64_t, VertexSample>& samples_by_vertex_id,
    RefinementTarget target,
    double minimum_curvature_sum) {
    if (target == RefinementTarget::none) {
        return false;
    }

    std::array<double, 4> curvature_sums;
    for (std::size_t index = 0; index < tet_vertices.size(); ++index) {
        const OrderedEigenSystem& eigen =
            samples_by_vertex_id.at(tet_vertices[index].value_of()).eigensystem;
        curvature_sums[index] = eigen.values[0] + eigen.values[2];
    }
    const int convexity = sign_consistency(curvature_sums);
    const bool ridge_is_strong = minimum_curvature_sum > 0.0
        ? std::all_of(curvature_sums.begin(), curvature_sums.end(), [minimum_curvature_sum](double value) {
              return value > minimum_curvature_sum;
          })
        : convexity >= 0;
    const bool valley_is_strong = minimum_curvature_sum > 0.0
        ? std::all_of(curvature_sums.begin(), curvature_sums.end(), [minimum_curvature_sum](double value) {
              return value < -minimum_curvature_sum;
          })
        : convexity <= 0;
    const bool ridge_crossing = ridge_is_strong &&
        has_condition_crossing(tet_vertices, samples_by_vertex_id, 0);
    const bool valley_crossing = valley_is_strong &&
        has_condition_crossing(tet_vertices, samples_by_vertex_id, 2);

    switch (target) {
    case RefinementTarget::ridges:
        return ridge_crossing;
    case RefinementTarget::valleys:
        return valley_crossing;
    case RefinementTarget::ridges_and_valleys:
        return ridge_crossing || valley_crossing;
    case RefinementTarget::none:
        return false;
    }
    return false;
}

bool tet_directions_are_orientable(
    std::span<const mtet::VertexId, 4> tet_vertices,
    const std::unordered_map<std::uint64_t, VertexSample>& samples_by_vertex_id,
    int direction_index) {
    std::array<Vec3, 4> directions;
    for (std::size_t index = 0; index < directions.size(); ++index) {
        directions[index] = samples_by_vertex_id.at(tet_vertices[index].value_of())
                                .eigensystem.vectors[direction_index];
        if (index > 0 && dot(directions[0], directions[index]) < 0.0) {
            directions[index] = directions[index] * -1.0;
        }
    }
    // Matching every direction to vertex 0 alone is insufficient: the other
    // three must also agree with one another. Do not mutate cached samples.
    return dot(directions[1], directions[2]) >= 0.0 &&
           dot(directions[1], directions[3]) >= 0.0 &&
           dot(directions[2], directions[3]) >= 0.0;
}

struct LinearityError {
    double maximum_difference = 0.0;
    double maximum_absolute_value = 0.0;
};

LinearityError evaluate_directional_linearity(
    const DifferentialField3D& field,
    const std::array<Vec3, 4>& positions,
    const std::array<const VertexSample*, 4>& vertex_samples,
    int direction_index,
    LinearitySampling sampling) {
    const Vec3 reference = vertex_samples[0]->eigensystem.vectors[direction_index];
    const auto condition_value = [&](const VertexSample& sample) {
        Vec3 direction = sample.eigensystem.vectors[direction_index];
        if (dot(direction, reference) < 0.0) {
            direction = direction * -1.0;
        }
        return dot(sample.gradient, direction);
    };
    LinearityError result;
    std::array<double, 4> vertex_values;
    for (std::size_t index = 0; index < 4; ++index) {
        vertex_values[index] = condition_value(*vertex_samples[index]);
        result.maximum_absolute_value = std::max(
            result.maximum_absolute_value, std::abs(vertex_values[index]));
    }
    const auto compare_sample = [&](const std::array<double, 4>& weights) {
        Vec3 position{};
        double linear_value = 0.0;
        for (std::size_t index = 0; index < 4; ++index) {
            position += positions[index] * weights[index];
            linear_value += vertex_values[index] * weights[index];
        }
        const double actual_value = condition_value(evaluate_vertex_sample(field, position));
        if (!std::isfinite(actual_value) || !std::isfinite(linear_value)) {
            throw std::runtime_error("non-finite directional linearity sample");
        }
        result.maximum_difference = std::max(
            result.maximum_difference, std::abs(actual_value - linear_value));
        result.maximum_absolute_value = std::max(
            result.maximum_absolute_value, std::abs(actual_value));
    };

    if (sampling == LinearitySampling::cubic_16) {
        // The twelve ordered vertex pairs are the two cubic locations per edge.
        for (std::size_t first = 0; first < 4; ++first) {
            for (std::size_t second = 0; second < 4; ++second) {
                if (first == second) continue;
                std::array<double, 4> weights{};
                weights[first] = 2.0 / 3.0;
                weights[second] = 1.0 / 3.0;
                compare_sample(weights);
            }
        }
    }
    for (std::size_t opposite_vertex = 0; opposite_vertex < 4; ++opposite_vertex) {
        std::array<double, 4> weights{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
        weights[opposite_vertex] = 0.0;
        compare_sample(weights);
    }
    if (sampling == LinearitySampling::face_centers_5) {
        compare_sample({.25, .25, .25, .25});
    }
    return result;
}

bool should_refine_tet(
    const mtet::MTetMesh& mesh,
    const DifferentialField3D& field,
    std::span<const mtet::VertexId, 4> tet_vertices,
    const std::unordered_map<std::uint64_t, VertexSample>& samples_by_vertex_id,
    const LongestEdgeRefinementOptions& options,
    double minimum_curvature_sum) {
    if (options.pipeline == RefinementPipeline::legacy_crossing) {
        return should_refine_tet_legacy(
            tet_vertices, samples_by_vertex_id, options.target, minimum_curvature_sum);
    }
    if (options.target == RefinementTarget::none) {
        return false;
    }

    bool all_positive = true;
    bool all_negative = true;
    bool ridge_is_strong = true;
    bool valley_is_strong = true;
    bool any_ridge_is_strong = false;
    bool any_valley_is_strong = false;
    for (const mtet::VertexId vertex_id : tet_vertices) {
        const double sum = samples_by_vertex_id.at(vertex_id.value_of()).curvature_sum;
        all_positive = all_positive && sum > options.curvature_zero_tolerance;
        all_negative = all_negative && sum < -options.curvature_zero_tolerance;
        ridge_is_strong = ridge_is_strong && sum > minimum_curvature_sum;
        valley_is_strong = valley_is_strong && sum < -minimum_curvature_sum;
        any_ridge_is_strong = any_ridge_is_strong ||
            sum > std::max(minimum_curvature_sum, options.curvature_zero_tolerance);
        any_valley_is_strong = any_valley_is_strong ||
            sum < -std::max(minimum_curvature_sum, options.curvature_zero_tolerance);
    }

    // Boundary II: mixed signs or an uncertain near-zero vertex. This is
    // independent of the directional crossing and the extraction threshold.
    const bool curvature_boundary = !all_positive && !all_negative;
    if (options.refine_curvature_boundary && curvature_boundary) {
        return true;
    }

    const bool wants_ridges = options.target == RefinementTarget::ridges ||
                              options.target == RefinementTarget::ridges_and_valleys;
    const bool wants_valleys = options.target == RefinementTarget::valleys ||
                               options.target == RefinementTarget::ridges_and_valleys;

    const auto needs_direction_refinement = [&](int direction_index, bool strong_enough) {
        // Boundary I remains an independent refinement trigger. For orientable
        // tets, linearity is only the final gate on an eligible crossing.
        if (!tet_directions_are_orientable(tet_vertices, samples_by_vertex_id, direction_index)) {
            return true;
        }
        if (!strong_enough || !has_condition_crossing(tet_vertices, samples_by_vertex_id, direction_index)) {
            return false;
        }
        if (options.linearity_sampling == LinearitySampling::disabled) {
            return true;
        }

        std::array<Vec3, 4> positions;
        std::array<const VertexSample*, 4> samples;
        for (std::size_t index = 0; index < 4; ++index) {
            positions[index] = position_of(mesh, tet_vertices[index]);
            samples[index] = &samples_by_vertex_id.at(tet_vertices[index].value_of());
        }
        const LinearityError error = evaluate_directional_linearity(
            field, positions, samples, direction_index, options.linearity_sampling);
        const double tolerance = options.linearity_absolute_tolerance +
            options.linearity_relative_tolerance * error.maximum_absolute_value;
        return error.maximum_difference > tolerance;
    };
    return (wants_ridges && !all_negative && needs_direction_refinement(
                0, curvature_boundary ? any_ridge_is_strong : ridge_is_strong)) ||
           (wants_valleys && !all_positive && needs_direction_refinement(
                2, curvature_boundary ? any_valley_is_strong : valley_is_strong));
}

std::optional<RefinementCandidate> make_refinement_candidate(
    const mtet::MTetMesh& mesh,
    const DifferentialField3D& field,
    mtet::TetId tet_id,
    const std::unordered_map<std::uint64_t, VertexSample>& samples_by_vertex_id,
    const LongestEdgeRefinementOptions& options,
    double minimum_curvature_sum) {
    constexpr std::array<std::array<int, 2>, 6> kLocalEdges{{
        {{0, 1}}, {{1, 2}}, {{2, 0}}, {{0, 3}}, {{1, 3}}, {{2, 3}},
    }};
    const auto tet_vertices = mesh.get_tet(tet_id);
    RefinementCandidate candidate{tet_id, 0, -1.0};
    for (std::uint8_t edge_index = 0; edge_index < kLocalEdges.size(); ++edge_index) {
        const auto [first_index, second_index] = kLocalEdges[edge_index];
        const double edge_length = norm(
            position_of(mesh, tet_vertices[first_index]) - position_of(mesh, tet_vertices[second_index]));
        if (edge_length > candidate.longest_edge_length) {
            candidate.longest_edge_length = edge_length;
            candidate.longest_local_edge = edge_index;
        }
    }
    // Avoid all additional derivative queries for already-small tetrahedra.
    if (candidate.longest_edge_length <= options.minimum_edge_length ||
        !should_refine_tet(mesh, field, tet_vertices, samples_by_vertex_id, options, minimum_curvature_sum)) {
        return std::nullopt;
    }
    return candidate;
}

int refine_longest_edges_in_place(
    mtet::MTetMesh& mesh,
    const DifferentialField3D& field,
    const LongestEdgeRefinementOptions& options,
    std::unordered_map<std::uint64_t, VertexSample>& samples_by_vertex_id,
    double minimum_curvature_sum) {
    if (options.target == RefinementTarget::none || options.max_splits == 0) {
        return 0;
    }

    // The ordered set is a max-priority queue with efficient removal by tet ID.
    // A split replaces only the tets around its edge, so all other candidates
    // remain valid and never need to be scanned again.
    using CandidateQueue = std::set<RefinementCandidate, LongestCandidateFirst>;
    CandidateQueue candidates;
    std::unordered_map<std::uint64_t, CandidateQueue::iterator> candidate_by_tet;

    // MTet's edge-ring traversal may return tagged tetrahedron handles.
    // Strip the tag before using a handle as a queue identity.
    const auto canonical_tet = [&](mtet::TetId tet_id) {
        return mesh.get_edge_tet(mesh.get_edge(tet_id, 0));
    };

    const auto enqueue_if_eligible = [&](mtet::TetId tet_id) {
        tet_id = canonical_tet(tet_id);
        if (candidate_by_tet.contains(tet_id.value_of())) {
            return;
        }
        const auto candidate = make_refinement_candidate(
            mesh, field, tet_id, samples_by_vertex_id, options, minimum_curvature_sum);
        if (!candidate || candidate->longest_edge_length <= options.minimum_edge_length) {
            return;
        }
        const auto [iterator, inserted] = candidates.insert(*candidate);
        if (inserted) {
            candidate_by_tet.emplace(tet_id.value_of(), iterator);
        }
    };

    const auto remove_candidate = [&](mtet::TetId tet_id) {
        tet_id = canonical_tet(tet_id);
        const auto entry = candidate_by_tet.find(tet_id.value_of());
        if (entry != candidate_by_tet.end()) {
            candidates.erase(entry->second);
            candidate_by_tet.erase(entry);
        }
    };

    mesh.seq_foreach_tet([&](mtet::TetId tet_id, auto) {
        enqueue_if_eligible(tet_id);
    });

    int split_count = 0;
    while (split_count < options.max_splits && !candidates.empty()) {
        const RefinementCandidate candidate = *candidates.begin();
        const mtet::EdgeId edge_id = mesh.get_edge(candidate.tet_id, candidate.longest_local_edge);

        std::vector<mtet::TetId> replaced_tets;
        mesh.foreach_tet_around_edge(edge_id, [&](mtet::TetId tet_id) {
            replaced_tets.push_back(tet_id);
        });
        for (const mtet::TetId tet_id : replaced_tets) {
            remove_candidate(tet_id);
        }

        const auto [new_vertex_id, first_half_edge, second_half_edge] = mesh.split_edge(edge_id);
        add_vertex_sample(mesh, field, new_vertex_id, samples_by_vertex_id);

        mesh.foreach_tet_around_edge(first_half_edge, enqueue_if_eligible);
        mesh.foreach_tet_around_edge(second_half_edge, enqueue_if_eligible);
        ++split_count;
    }
    return split_count;
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

double standalone_linearity_error(
    const DifferentialField3D& field, const std::array<Vec3, 4>& positions,
    int direction_index, LinearitySampling sampling) {
    if (!field.gradient || !field.hessian || direction_index < 0 || direction_index > 2) {
        throw std::invalid_argument("linearity test requires derivatives and direction_index in [0,2]");
    }
    std::array<VertexSample, 4> samples;
    std::array<const VertexSample*, 4> sample_pointers;
    for (std::size_t index = 0; index < 4; ++index) {
        samples[index] = evaluate_vertex_sample(field, positions[index]);
        sample_pointers[index] = &samples[index];
    }
    return evaluate_directional_linearity(
        field, positions, sample_pointers, direction_index, sampling).maximum_difference;
}

} // namespace

double directional_linearity_error_16(
    const DifferentialField3D& field, const std::array<Vec3, 4>& tet_vertices,
    int direction_index) {
    return standalone_linearity_error(field, tet_vertices, direction_index, LinearitySampling::cubic_16);
}

double directional_linearity_error_5(
    const DifferentialField3D& field, const std::array<Vec3, 4>& tet_vertices,
    int direction_index) {
    return standalone_linearity_error(field, tet_vertices, direction_index, LinearitySampling::face_centers_5);
}

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
    coarse_grid.seq_foreach_vertex([&](mtet::VertexId vertex_id, auto position) {
        samples_by_vertex_id.emplace(
            vertex_id.value_of(), evaluate_vertex_sample(field, {position[0], position[1], position[2]}));
    });

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

    for (const GridEdge& edge : build_edge_adjacency(coarse_grid)) {
        const VertexSample& first_sample = samples_by_vertex_id.at(edge.first_vertex.value_of());
        const VertexSample& second_sample = samples_by_vertex_id.at(edge.second_vertex.value_of());
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
        if (!ridge && !valley) continue;

        const auto append_crossing = [&](bool is_ridge) {
            const int label = is_ridge ? 1 : -1;
            const int direction_index = is_ridge ? 0 : 2;
            Vec3 crossing_point;
            const bool found_crossing = options.subdivide_roots
                ? find_refined_crossing(field, position_of(coarse_grid, edge.first_vertex),
                      position_of(coarse_grid, edge.second_vertex), first_sample.gradient,
                      second_sample.gradient, first_sample.eigensystem.vectors[direction_index],
                      second_sample.eigensystem.vectors[direction_index], is_ridge, options, crossing_point)
                : find_linear_crossing(position_of(coarse_grid, edge.first_vertex),
                      position_of(coarse_grid, edge.second_vertex), first_sample.gradient,
                      second_sample.gradient, first_sample.eigensystem.vectors[direction_index],
                      second_sample.eigensystem.vectors[direction_index], crossing_point);
            if (!found_crossing) return false;
            if (is_ridge && check_ridge_at_crossing) {
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

            crossing_edges.push_back({edge.first_vertex, edge.second_vertex, edge.incident_tets, label});
            for (const std::size_t tet_index : edge.incident_tets) {
                crossing_sum_by_tet[tet_index] += crossing_point;
                ++crossing_count_by_tet[tet_index];
            }
            return true;
        };
        if (ridge) {
            const bool accepted = append_crossing(true);
            // The broader ridge precheck must not hide a valid valley when
            // its actual root fails the ridge test. Original mode is unchanged.
            if (!accepted && check_ridge_at_crossing && valley) append_crossing(false);
        } else if (valley) {
            append_crossing(false);
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
