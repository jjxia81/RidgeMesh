#pragma once

#include "ridge_surface/ridge_surface.hpp"

#include <cstdint>
#include <unordered_map>

namespace ridge_surface::detail {

// Shared by adaptive refinement and the final edge-crossing extraction.
struct OrderedEigenSystem {
    std::array<double, 3> values;
    std::array<Vec3, 3> vectors;
};

struct VertexSample {
    Vec3 gradient;
    OrderedEigenSystem eigensystem;
    double curvature_sum;
};

OrderedEigenSystem compute_eigensystem(const Mat3& matrix);
Vec3 position_of(const mtet::MTetMesh& mesh, mtet::VertexId vertex_id);
VertexSample evaluate_vertex_sample(const DifferentialField3D& field, Vec3 point);

int refine_longest_edges_in_place(
    mtet::MTetMesh& mesh,
    const DifferentialField3D& field,
    const LongestEdgeRefinementOptions& options,
    std::unordered_map<std::uint64_t, VertexSample>& samples_by_vertex_id,
    double minimum_curvature_sum);

} // namespace ridge_surface::detail
