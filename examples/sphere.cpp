#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

#include <mtet/grid.h>

#include "ridge_surface/ridge_surface.hpp"

namespace {

void write_surface_ply(const ridge_surface::SurfaceMesh& mesh, const char* filename) {
    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("could not open surface PLY for writing");
    }

    const std::size_t face_count = mesh.ridge_triangles.size() + mesh.valley_triangles.size();
    output << "ply\nformat ascii 1.0\n"
           << "element vertex " << mesh.vertices.size() << "\n"
           << "property float x\nproperty float y\nproperty float z\n"
           << "element face " << face_count << "\n"
           << "property list uchar int vertex_indices\nend_header\n";
    for (const ridge_surface::Vec3& point : mesh.vertices) {
        output << point.x << ' ' << point.y << ' ' << point.z << '\n';
    }

    const auto write_triangles = [&](const std::vector<ridge_surface::Triangle>& triangles) {
        for (const ridge_surface::Triangle& triangle : triangles) {
            output << "3 " << triangle.indices[0] << ' ' << triangle.indices[1] << ' '
                   << triangle.indices[2] << '\n';
        }
    };
    write_triangles(mesh.ridge_triangles);
    write_triangles(mesh.valley_triangles);
}

// PLY has no universal tetrahedron element. This vertex/edge PLY makes every
// edge created by MTet's adaptive bisection visible as a wireframe.
void write_grid_wireframe_ply(mtet::MTetMesh& grid, const char* filename) {
    std::map<std::uint64_t, std::size_t> output_vertex_by_mtet_id;
    std::vector<std::array<double, 3>> vertices;
    grid.seq_foreach_vertex([&](mtet::VertexId vertex_id, auto point) {
        output_vertex_by_mtet_id.emplace(vertex_id.value_of(), vertices.size());
        vertices.push_back({point[0], point[1], point[2]});
    });

    std::map<std::uint64_t, std::pair<mtet::VertexId, mtet::VertexId>> edges;
    grid.seq_foreach_tet([&](mtet::TetId tet_id, auto) {
        grid.foreach_edge_in_tet(tet_id, [&](mtet::EdgeId edge_id, mtet::VertexId first, mtet::VertexId second) {
            edges.emplace(edge_id.value_of(), std::make_pair(first, second));
        });
    });

    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("could not open adaptive-grid PLY for writing");
    }
    output << "ply\nformat ascii 1.0\n"
           << "element vertex " << vertices.size() << "\n"
           << "property float x\nproperty float y\nproperty float z\n"
           << "element edge " << edges.size() << "\n"
           << "property int vertex1\nproperty int vertex2\nend_header\n";
    for (const auto& point : vertices) {
        output << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
    }
    for (const auto& [edge_id, endpoints] : edges) {
        (void)edge_id;
        output << output_vertex_by_mtet_id.at(endpoints.first.value_of()) << ' '
               << output_vertex_by_mtet_id.at(endpoints.second.value_of()) << '\n';
    }
}

} // namespace

int main() {
    using namespace ridge_surface;

    constexpr double sphere_radius = 0.6;
    // f = -(||x||^2 - r^2)^2 has its maximum, and a strong height ridge,
    // on the sphere ||x|| = r. It is polynomial, so unlike an unsigned
    // distance field it is differentiable everywhere.
    DifferentialField3D field{
        [=](const Vec3& point) {
            const double offset = dot(point, point) - sphere_radius * sphere_radius;
            return point * (-4.0 * offset);
        },
        [=](const Vec3& point) {
            const double offset = dot(point, point) - sphere_radius * sphere_radius;
            const std::array<double, 3> coordinates{point.x, point.y, point.z};
            Mat3 hessian{};
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    hessian[row][column] = -8.0 * coordinates[row] * coordinates[column];
                    if (row == column) {
                        hessian[row][column] -= 4.0 * offset;
                    }
                }
            }
            return hessian;
        },
    };

    // This is intentionally coarse. The surfacer refines this MTet grid in place.
    // Offset x by 0.05 so no symmetry plane passes exactly through all samples.
    mtet::MTetMesh adaptive_grid = mtet::generate_tet_grid(
        {4, 4, 4}, {-1.05, -1, -1}, {0.95, 1, 1}, mtet::TET6);
    const std::size_t coarse_vertex_count = adaptive_grid.get_num_vertices();
    const std::size_t coarse_tet_count = adaptive_grid.get_num_tets();

    SurfaceOptions options;
    options.surface_target = RefinementTarget::ridges;
    options.longest_edge_refinement.target = RefinementTarget::ridges;
    options.longest_edge_refinement.max_splits = 300;
    options.longest_edge_refinement.minimum_edge_length = 0.025;

    const SurfaceMesh surface = extract_height_ridges(field, adaptive_grid, options);
    write_surface_ply(surface, "ridge_example.ply");
    write_grid_wireframe_ply(adaptive_grid, "adaptive_grid_wireframe.ply");

    std::cout << "coarse grid: " << coarse_vertex_count << " vertices, " << coarse_tet_count << " tetrahedra\n"
              << "adaptive grid: " << adaptive_grid.get_num_vertices() << " vertices, "
              << adaptive_grid.get_num_tets() << " tetrahedra\n"
              << "surface: " << surface.vertices.size() << " dual vertices, "
              << surface.ridge_triangles.size() << " ridge triangles, "
              << surface.valley_triangles.size() << " valley triangles\n"
              << "wrote ridge_example.ply and adaptive_grid_wireframe.ply\n";
}
