#include "adaptive_refine.hpp"

#include <Eigen/Eigenvalues>
#include <nanothread/nanothread.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ridge_surface::detail {

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

namespace {

struct RefinementCandidate {
    mtet::TetId tet_id;
    std::uint8_t longest_local_edge;
    double longest_edge_length;
};

// The Bezier-simplex predicate uses points shared by neighboring tetrahedra:
// two trisection points on every edge and one center on every triangular face.
// Cache those by their vertex IDs so a point is evaluated only once, even when
// several candidate tetrahedra reference it.
struct BezierSampleKey {
    std::array<std::uint64_t, 3> vertices{};
    std::uint8_t kind = 0; // 1/2: (2*a+b)/3 or (a+2*b)/3; 3: face center

    bool operator==(const BezierSampleKey&) const = default;
};

struct BezierSampleKeyHash {
    std::size_t operator()(const BezierSampleKey& key) const noexcept {
        std::size_t result = key.kind;
        for (const std::uint64_t vertex : key.vertices)
            result ^= std::hash<std::uint64_t>{}(vertex) + 0x9e3779b9 + (result << 6) + (result >> 2);
        return result;
    }
};

using BezierSampleCache = std::unordered_map<BezierSampleKey, VertexSample, BezierSampleKeyHash>;

BezierSampleKey edge_third_key(mtet::VertexId first, mtet::VertexId second, std::uint8_t kind) {
    std::uint64_t low = first.value_of();
    std::uint64_t high = second.value_of();
    if (low > high) {
        std::swap(low, high);
        // Keep the physical point unchanged after canonicalizing the edge.
        kind = kind == 1 ? 2 : 1;
    }
    return {{low, high, 0}, kind};
}

BezierSampleKey face_center_key(mtet::VertexId first, mtet::VertexId second, mtet::VertexId third) {
    std::array<std::uint64_t, 3> vertices{first.value_of(), second.value_of(), third.value_of()};
    std::sort(vertices.begin(), vertices.end());
    return {vertices, 3};
}

Vec3 position_for_bezier_key(const mtet::MTetMesh& mesh, const BezierSampleKey& key) {
    const auto first = position_of(mesh, mtet::VertexId(key.vertices[0]));
    const auto second = position_of(mesh, mtet::VertexId(key.vertices[1]));
    if (key.kind == 1) return (2.0 * first + second) / 3.0;
    if (key.kind == 2) return (first + 2.0 * second) / 3.0;
    const auto third = position_of(mesh, mtet::VertexId(key.vertices[2]));
    return (first + second + third) / 3.0;
}

struct LongestCandidateFirst {
    bool operator()(const RefinementCandidate& left, const RefinementCandidate& right) const {
        if (left.longest_edge_length != right.longest_edge_length) {
            return left.longest_edge_length > right.longest_edge_length;
        }
        return left.tet_id.value_of() < right.tet_id.value_of();
    }
};

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

bool has_edge_condition_crossing(
    std::span<const mtet::VertexId, 4> tet_vertices,
    const std::unordered_map<std::uint64_t, VertexSample>& samples_by_vertex_id,
    int direction_index) {
    // Check all six edges independently. No shared tet orientation is assumed.
    for (std::size_t first = 0; first < 4; ++first) {
        const VertexSample& a = samples_by_vertex_id.at(tet_vertices[first].value_of());
        const Vec3 first_direction = a.eigensystem.vectors[direction_index];
        const double first_value = dot(a.gradient, first_direction);
        for (std::size_t second = first + 1; second < 4; ++second) {
            const VertexSample& b = samples_by_vertex_id.at(tet_vertices[second].value_of());
            Vec3 second_direction = b.eigensystem.vectors[direction_index];
            if (dot(first_direction, second_direction) < 0.0) {
                second_direction = second_direction * -1.0;
            }
            const double second_value = dot(b.gradient, second_direction);
            // Match surfacing's strict sign-change test, excluding zero edges.
            if ((first_value < 0.0 && second_value > 0.0) ||
                (first_value > 0.0 && second_value < 0.0)) {
                return true;
            }
        }
    }
    return false;
}

bool eigenvectors_exceed_variation(
    std::span<const VertexSample> samples, int direction_index, double angle_limit) {
    // Average axes rather than signed eigenvectors: v and -v contribute the
    // same outer product. The dominant eigenvector of this scatter matrix is
    // the local mean direction without any manual sign synchronization.
    Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
    for (const VertexSample& sample : samples) {
        const Vec3& direction = sample.eigensystem.vectors[direction_index];
        const Eigen::Vector3d v(direction.x, direction.y, direction.z);
        scatter.noalias() += v * v.transpose();
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scatter);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("eigenvector-variation mean-axis solve failed");
    const Eigen::Vector3d mean_axis = solver.eigenvectors().col(2);
    const double cosine_limit = std::cos(angle_limit);
    for (const VertexSample& sample : samples) {
        const Vec3& direction = sample.eigensystem.vectors[direction_index];
        const double alignment = std::abs(
            direction.x * mean_axis.x() + direction.y * mean_axis.y() + direction.z * mean_axis.z());
        // acos(alignment) > angle_limit iff alignment < cos(angle_limit).
        if (alignment < cosine_limit) return true;
    }
    return false;
}

bool tet_eigenvectors_exceed_variation(
    std::span<const mtet::VertexId, 4> tet_vertices,
    const std::unordered_map<std::uint64_t, VertexSample>& samples_by_vertex_id,
    int direction_index, double angle_limit) {
    std::array<VertexSample, 4> samples;
    for (std::size_t index = 0; index < samples.size(); ++index)
        samples[index] = samples_by_vertex_id.at(tet_vertices[index].value_of());
    return eigenvectors_exceed_variation(samples, direction_index, angle_limit);
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

std::array<std::array<double, 4>, 16> cubic_sample_weights() {
    std::array<std::array<double, 4>, 16> weights{};
    std::size_t sample_index = 0;
    for (std::size_t first = 0; first < 4; ++first) {
        for (std::size_t second = 0; second < 4; ++second) {
            if (first == second) continue;
            weights[sample_index][first] = 2.0 / 3.0;
            weights[sample_index++][second] = 1.0 / 3.0;
        }
    }
    for (std::size_t opposite = 0; opposite < 4; ++opposite) {
        weights[sample_index].fill(1.0 / 3.0);
        weights[sample_index++][opposite] = 0.0;
    }
    return weights;
}

bool should_refine_bezier_simplex(
    const mtet::MTetMesh& mesh,
    const DifferentialField3D& field,
    std::span<const mtet::VertexId, 4> tet_vertices,
    const std::unordered_map<std::uint64_t, VertexSample>& samples_by_vertex_id,
    const BezierSampleCache& bezier_samples,
    const LongestEdgeRefinementOptions& options) {
    // These are sampled values at cubic barycentric locations, not Bernstein
    // coefficients. Their edge/face keys are shared across adjacent tets.
    std::array<VertexSample, 20> samples;
    for (std::size_t index = 0; index < 4; ++index) {
        samples[index] = samples_by_vertex_id.at(tet_vertices[index].value_of());
    }
    std::size_t sample_index = 4;
    for (std::size_t first = 0; first < 4; ++first)
        for (std::size_t second = 0; second < 4; ++second)
            if (first != second)
                samples[sample_index++] = bezier_samples.at(
                    edge_third_key(tet_vertices[first], tet_vertices[second], 1));
    for (std::size_t opposite = 0; opposite < 4; ++opposite) {
        std::array<mtet::VertexId, 3> face;
        std::size_t face_index = 0;
        for (std::size_t vertex = 0; vertex < 4; ++vertex)
            if (vertex != opposite) face[face_index++] = tet_vertices[vertex];
        samples[sample_index++] = bezier_samples.at(face_center_key(face[0], face[1], face[2]));
    }

    const auto selected_surface_needs_refinement = [&](bool ridge) {
        double minimum_sum = std::numeric_limits<double>::infinity();
        double maximum_sum = -std::numeric_limits<double>::infinity();
        for (const VertexSample& sample : samples) {
            const double signed_sum = ridge ? sample.curvature_sum : -sample.curvature_sum;
            if (!std::isfinite(signed_sum)) {
                throw std::runtime_error("non-finite Bezier-simplex curvature sample");
            }
            minimum_sum = std::min(minimum_sum, signed_sum);
            maximum_sum = std::max(maximum_sum, signed_sum);
        }
        // 1. A positive curvature sum anywhere among the 20 locations.
        if (!(maximum_sum > 0.0)) return false;
        const int direction_index = ridge ? 0 : 2;
        // 2. A strict crossing on any of the six original tet edges, using
        // independent pair orientation. Sample locations do not add grid edges.
        if (!has_edge_condition_crossing(tet_vertices, samples_by_vertex_id, direction_index)) {
            return false;
        }
        // 3. Curvature boundary, now tested on all 20 sampled values.
        if (minimum_sum <= options.curvature_zero_tolerance &&
            maximum_sum >= -options.curvature_zero_tolerance) {
            return true;
        }
        // 4. Tet-wide orientability. Unlike the older pipeline these two
        // boundary checks are intrinsic to this explicitly selected pipeline.
        if (!tet_directions_are_orientable(tet_vertices, samples_by_vertex_id, direction_index)) {
            return true;
        }
        // 5. The selected final gate decides whether this eligible tet needs
        // refinement. Eigenvector variation is evaluated on the actual tet
        // vertices and is therefore independent of the interior samples.
        const bool use_linearity = options.refinement_gate == RefinementGate::linearity ||
                                   options.refinement_gate == RefinementGate::either;
        const bool use_eigenvector_variation =
            options.refinement_gate == RefinementGate::eigenvector_variation ||
            options.refinement_gate == RefinementGate::either;
        const bool eigenvector_varies = use_eigenvector_variation &&
            eigenvectors_exceed_variation(samples, direction_index,
                                          options.eigenvector_variation_angle);
        if (!use_linearity) return eigenvector_varies;
        if (options.linearity_sampling == LinearitySampling::disabled) return true;
        // Linearity reuses the 16 samples already evaluated.
        const Vec3 reference = samples[0].eigensystem.vectors[direction_index];
        const auto condition_value = [&](const VertexSample& sample) {
            Vec3 direction = sample.eigensystem.vectors[direction_index];
            if (dot(direction, reference) < 0.0) direction = direction * -1.0;
            return dot(sample.gradient, direction);
        };
        LinearityError error;
        std::array<double, 4> vertex_values;
        for (std::size_t index = 0; index < 4; ++index) {
            vertex_values[index] = condition_value(samples[index]);
            error.maximum_absolute_value = std::max(
                error.maximum_absolute_value, std::abs(vertex_values[index]));
        }
        const auto compare = [&](const VertexSample& sample, const std::array<double, 4>& barycentric) {
            double linear_value = 0.0;
            for (std::size_t index = 0; index < 4; ++index) {
                linear_value += vertex_values[index] * barycentric[index];
            }
            const double actual_value = condition_value(sample);
            if (!std::isfinite(actual_value) || !std::isfinite(linear_value)) {
                throw std::runtime_error("non-finite Bezier-simplex condition sample");
            }
            error.maximum_difference = std::max(error.maximum_difference, std::abs(actual_value - linear_value));
            error.maximum_absolute_value = std::max(error.maximum_absolute_value, std::abs(actual_value));
        };
        const auto weights = cubic_sample_weights();
        const std::size_t first_sample = options.linearity_sampling == LinearitySampling::face_centers_5 ? 12 : 0;
        for (std::size_t index = first_sample; index < weights.size(); ++index)
            compare(samples[index + 4], weights[index]);
        if (options.linearity_sampling == LinearitySampling::face_centers_5) {
            // The tetrahedron center is not shared, and is used only by this
            // optional five-point linearity variant.
            Vec3 center{};
            for (const mtet::VertexId vertex : tet_vertices) center += position_of(mesh, vertex);
            compare(evaluate_vertex_sample(field, center), {.25, .25, .25, .25});
        }
        const bool non_linear = error.maximum_difference > options.linearity_absolute_tolerance +
            options.linearity_relative_tolerance * error.maximum_absolute_value;
        return non_linear || eigenvector_varies;
    };
    const bool ridges = options.target == RefinementTarget::ridges ||
                        options.target == RefinementTarget::ridges_and_valleys;
    const bool valleys = options.target == RefinementTarget::valleys ||
                         options.target == RefinementTarget::ridges_and_valleys;
    return (ridges && selected_surface_needs_refinement(true)) ||
           (valleys && selected_surface_needs_refinement(false));
}

bool should_refine_tet(
    const mtet::MTetMesh& mesh,
    const DifferentialField3D& field,
    std::span<const mtet::VertexId, 4> tet_vertices,
    const std::unordered_map<std::uint64_t, VertexSample>& samples_by_vertex_id,
    const BezierSampleCache& bezier_samples,
    const LongestEdgeRefinementOptions& options,
    double minimum_curvature_sum) {
    if (options.pipeline == RefinementPipeline::legacy_crossing) {
        return should_refine_tet_legacy(
            tet_vertices, samples_by_vertex_id, options.target, minimum_curvature_sum);
    }
    if (options.target == RefinementTarget::none) {
        return false;
    }

    if (options.pipeline == RefinementPipeline::bezier_simplex) {
        return should_refine_bezier_simplex(mesh, field, tet_vertices, samples_by_vertex_id, bezier_samples, options);
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
        if (options.refine_orientation_boundary &&
            !tet_directions_are_orientable(tet_vertices, samples_by_vertex_id, direction_index)) {
            return true;
        }
        if (!strong_enough) {
            return false;
        }
        const bool crossing = options.refine_orientation_boundary
            ? has_condition_crossing(tet_vertices, samples_by_vertex_id, direction_index)
            : has_edge_condition_crossing(tet_vertices, samples_by_vertex_id, direction_index);
        if (!crossing) {
            return false;
        }
        const bool use_linearity = options.refinement_gate == RefinementGate::linearity ||
                                   options.refinement_gate == RefinementGate::either;
        const bool use_eigenvector_variation =
            options.refinement_gate == RefinementGate::eigenvector_variation ||
            options.refinement_gate == RefinementGate::either;
        const bool eigenvector_varies = use_eigenvector_variation &&
            tet_eigenvectors_exceed_variation(tet_vertices, samples_by_vertex_id,
                                               direction_index,
                                               options.eigenvector_variation_angle);
        if (!use_linearity) return eigenvector_varies;
        if (options.linearity_sampling == LinearitySampling::disabled) return true;

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
        return error.maximum_difference > tolerance || eigenvector_varies;
    };
    return (wants_ridges && !all_negative && needs_direction_refinement(
                0, curvature_boundary ? any_ridge_is_strong : ridge_is_strong)) ||
           (wants_valleys && !all_positive && needs_direction_refinement(
                2, curvature_boundary ? any_valley_is_strong : valley_is_strong));
}

std::optional<RefinementCandidate> make_refinement_candidate(
    const mtet::MTetMesh& mesh, mtet::TetId tet_id,
    const LongestEdgeRefinementOptions& options) {
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
    // Classification is deferred until a batch has collected and evaluated all
    // shared Bezier edge/face samples in parallel.
    if (candidate.longest_edge_length <= options.minimum_edge_length) return std::nullopt;
    return candidate;
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

int refine_longest_edges_in_place(
    mtet::MTetMesh& mesh,
    const DifferentialField3D& field,
    const LongestEdgeRefinementOptions& options,
    std::unordered_map<std::uint64_t, VertexSample>& samples_by_vertex_id,
    double minimum_curvature_sum) {
    if (options.target == RefinementTarget::none || options.max_splits == 0) {
        return 0;
    }

    // The ordered set is a max-priority queue of tetrahedra awaiting
    // classification. We classify bounded batches only after their shared
    // Bezier samples have been evaluated and cached.
    using CandidateQueue = std::set<RefinementCandidate, LongestCandidateFirst>;
    CandidateQueue candidates;
    std::unordered_map<std::uint64_t, CandidateQueue::iterator> candidate_by_tet;
    BezierSampleCache bezier_samples;

    const auto enqueue_if_eligible = [&](mtet::TetId tet_id) {
        if (!mesh.has_tet(tet_id)) return;
        if (candidate_by_tet.contains(tet_id.value_of())) {
            return;
        }
        const auto candidate = make_refinement_candidate(mesh, tet_id, options);
        if (!candidate || candidate->longest_edge_length <= options.minimum_edge_length) {
            return;
        }
        const auto [iterator, inserted] = candidates.insert(*candidate);
        if (inserted) {
            candidate_by_tet.emplace(tet_id.value_of(), iterator);
        }
    };

    const auto remove_candidate = [&](mtet::TetId tet_id) {
        const auto entry = candidate_by_tet.find(tet_id.value_of());
        if (entry != candidate_by_tet.end()) {
            candidates.erase(entry->second);
            candidate_by_tet.erase(entry);
        }
    };

    mesh.seq_foreach_tet([&](mtet::TetId tet_id, auto) {
        enqueue_if_eligible(tet_id);
    });

    const auto add_bezier_key = [&](const BezierSampleKey& key,
                                    std::unordered_map<BezierSampleKey, Vec3, BezierSampleKeyHash>& missing) {
        if (!bezier_samples.contains(key)) missing.try_emplace(key, position_for_bezier_key(mesh, key));
    };
    const auto collect_bezier_samples = [&](const std::vector<RefinementCandidate>& batch) {
        std::unordered_map<BezierSampleKey, Vec3, BezierSampleKeyHash> missing;
        for (const RefinementCandidate& candidate : batch) {
            if (!mesh.has_tet(candidate.tet_id)) continue;
            const auto vertices = mesh.get_tet(candidate.tet_id);
            for (std::size_t first = 0; first < 4; ++first)
                for (std::size_t second = first + 1; second < 4; ++second) {
                    add_bezier_key(edge_third_key(vertices[first], vertices[second], 1), missing);
                    add_bezier_key(edge_third_key(vertices[first], vertices[second], 2), missing);
                }
            for (std::size_t opposite = 0; opposite < 4; ++opposite) {
                std::array<mtet::VertexId, 3> face;
                std::size_t face_index = 0;
                for (std::size_t vertex = 0; vertex < 4; ++vertex)
                    if (vertex != opposite) face[face_index++] = vertices[vertex];
                add_bezier_key(face_center_key(face[0], face[1], face[2]), missing);
            }
        }
        std::vector<std::pair<BezierSampleKey, Vec3>> requests;
        requests.reserve(missing.size());
        for (const auto& request : missing) requests.push_back(request);
        std::vector<VertexSample> evaluated(requests.size());
        std::exception_ptr evaluation_error;
        std::mutex error_mutex;
        drjit::parallel_for(drjit::blocked_range<std::size_t>(0, requests.size(), 1),
            [&](drjit::blocked_range<std::size_t> range) {
                for (std::size_t index = range.begin(); index != range.end(); ++index) {
                    try { evaluated[index] = evaluate_vertex_sample(field, requests[index].second); }
                    catch (...) { std::lock_guard lock(error_mutex); if (!evaluation_error) evaluation_error = std::current_exception(); }
                }
            });
        if (evaluation_error) std::rethrow_exception(evaluation_error);
        for (std::size_t index = 0; index < requests.size(); ++index)
            bezier_samples.emplace(requests[index].first, evaluated[index]);
    };

    int split_count = 0;
    while (split_count < options.max_splits && !candidates.empty()) {
        constexpr std::size_t kBatchSize = 512;
        std::vector<RefinementCandidate> batch;
        batch.reserve(kBatchSize);
        while (!candidates.empty() && batch.size() < kBatchSize) {
            const RefinementCandidate candidate = *candidates.begin();
            candidate_by_tet.erase(candidate.tet_id.value_of());
            candidates.erase(candidates.begin());
            batch.push_back(candidate);
        }
        if (options.pipeline == RefinementPipeline::bezier_simplex) collect_bezier_samples(batch);

        std::vector<bool> marked(batch.size(), false);
        std::exception_ptr classification_error;
        std::mutex error_mutex;
        drjit::parallel_for(drjit::blocked_range<std::size_t>(0, batch.size(), 1),
            [&](drjit::blocked_range<std::size_t> range) {
                for (std::size_t index = range.begin(); index != range.end(); ++index) {
                    try {
                        const RefinementCandidate& candidate = batch[index];
                        if (!mesh.has_tet(candidate.tet_id)) continue;
                        marked[index] = should_refine_tet(
                            mesh, field, mesh.get_tet(candidate.tet_id), samples_by_vertex_id,
                            bezier_samples, options, minimum_curvature_sum);
                    } catch (...) {
                        std::lock_guard lock(error_mutex);
                        if (!classification_error) classification_error = std::current_exception();
                    }
                }
            });
        if (classification_error) std::rethrow_exception(classification_error);

        std::vector<mtet::VertexId> new_vertices;
        std::vector<mtet::TetId> affected_tets;
        for (std::size_t index = 0; index < batch.size() && split_count < options.max_splits; ++index) {
            if (!marked[index]) continue;
            const RefinementCandidate& candidate = batch[index];
            // Earlier splits in this batch can replace this tet/edge. Revalidate
            // immediately before mutation; stale candidates are simply skipped.
            if (!mesh.has_tet(candidate.tet_id)) continue;
            const mtet::EdgeId edge_id = mesh.get_edge(candidate.tet_id, candidate.longest_local_edge);
            if (!mesh.has_edge(edge_id)) continue;
            std::vector<mtet::TetId> replaced_tets;
            mesh.foreach_tet_around_edge(edge_id, [&](mtet::TetId tet_id) { replaced_tets.push_back(tet_id); });
            for (const mtet::TetId tet_id : replaced_tets) remove_candidate(tet_id);
            const auto [new_vertex_id, first_half_edge, second_half_edge] = mesh.split_edge(edge_id);
            new_vertices.push_back(new_vertex_id);
            mesh.foreach_tet_around_edge(first_half_edge, [&](mtet::TetId tet_id) { affected_tets.push_back(tet_id); });
            mesh.foreach_tet_around_edge(second_half_edge, [&](mtet::TetId tet_id) { affected_tets.push_back(tet_id); });
            ++split_count;
        }
        std::vector<VertexSample> new_samples(new_vertices.size());
        drjit::parallel_for(drjit::blocked_range<std::size_t>(0, new_vertices.size(), 1),
            [&](drjit::blocked_range<std::size_t> range) {
                for (std::size_t index = range.begin(); index != range.end(); ++index)
                    new_samples[index] = evaluate_vertex_sample(field, position_of(mesh, new_vertices[index]));
            });
        for (std::size_t index = 0; index < new_vertices.size(); ++index)
            samples_by_vertex_id.try_emplace(new_vertices[index].value_of(), new_samples[index]);
        for (const mtet::TetId tet_id : affected_tets) enqueue_if_eligible(tet_id);
    }
    return split_count;
}

} // namespace ridge_surface::detail

namespace ridge_surface {

double directional_linearity_error_16(
    const DifferentialField3D& field, const std::array<Vec3, 4>& tet_vertices,
    int direction_index) {
    return detail::standalone_linearity_error(field, tet_vertices, direction_index, LinearitySampling::cubic_16);
}

double directional_linearity_error_5(
    const DifferentialField3D& field, const std::array<Vec3, 4>& tet_vertices,
    int direction_index) {
    return detail::standalone_linearity_error(field, tet_vertices, direction_index, LinearitySampling::face_centers_5);
}

} // namespace ridge_surface
