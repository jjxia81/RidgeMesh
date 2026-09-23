#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <mtet/grid.h>

#include "ridge_surface/ridge_surface.hpp"

namespace {

void write_ridge_ply(const ridge_surface::SurfaceMesh& mesh, const char* filename) {
    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("could not open ellipsoid PLY for writing");
    }

    output << "ply\nformat ascii 1.0\n"
           << "element vertex " << mesh.vertices.size() << "\n"
           << "property float x\nproperty float y\nproperty float z\n"
           << "element face " << mesh.ridge_triangles.size() << "\n"
           << "property list uchar int vertex_indices\nend_header\n";
    for (const ridge_surface::Vec3& point : mesh.vertices) {
        output << point.x << ' ' << point.y << ' ' << point.z << '\n';
    }
    for (const ridge_surface::Triangle& triangle : mesh.ridge_triangles) {
        output << "3 " << triangle.indices[0] << ' ' << triangle.indices[1] << ' '
               << triangle.indices[2] << '\n';
    }
}

} // namespace

int main() {
    using namespace ridge_surface;

    // The surface q(x)=1 is an ellipsoid. F=-((q-1)^2) is smooth and has a
    // maximum of zero along that surface, so it is suitable for height-ridge
    // extraction even though a raw UDF would have a derivative cusp there.
    // Equal horizontal radii make this a sphere compressed along z.
    constexpr std::array<double, 3> radii{0.75, 0.75, 0.25};
    constexpr std::array<double, 3> inverse_radius_squared{
        1.0 / (radii[0] * radii[0]),
        1.0 / (radii[1] * radii[1]),
        1.0 / (radii[2] * radii[2]),
    };

    const auto ellipsoid_q = [&](const Vec3& point) {
        return point.x * point.x * inverse_radius_squared[0]
             + point.y * point.y * inverse_radius_squared[1]
             + point.z * point.z * inverse_radius_squared[2];
    };

    DifferentialField3D field{
        [&](const Vec3& point) {
            const double offset = ellipsoid_q(point) - 1.0;
            return Vec3{
                -4.0 * offset * inverse_radius_squared[0] * point.x,
                -4.0 * offset * inverse_radius_squared[1] * point.y,
                -4.0 * offset * inverse_radius_squared[2] * point.z,
            };
        },
        [&](const Vec3& point) {
            const double offset = ellipsoid_q(point) - 1.0;
            const std::array<double, 3> coordinates{point.x, point.y, point.z};
            Mat3 hessian{};
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    hessian[row][column] = -8.0 * inverse_radius_squared[row]
                        * coordinates[row] * inverse_radius_squared[column] * coordinates[column];
                    if (row == column) {
                        hessian[row][column] -= 4.0 * offset * inverse_radius_squared[row];
                    }
                }
            }
            return hessian;
        },
    };

    // Offset the x bound so the ellipsoid does not coincide with a grid plane.
    mtet::MTetMesh grid = mtet::generate_tet_grid(
        {64, 64, 64}, {-0.98, -1.0, -0.40}, {1.02, 1.0, 0.40}, mtet::TET6);
    SurfaceOptions options;
    options.surface_target = RefinementTarget::ridges;

    const SurfaceMesh surface = extract_height_ridges(field, grid, options);
    if (surface.ridge_triangles.empty()) {
        throw std::runtime_error("ellipsoid example did not extract a ridge mesh");
    }
    write_ridge_ply(surface, "ellipsoid_ridge.ply");

    std::cout << "ellipsoid grid: " << grid.get_num_vertices() << " vertices, "
              << grid.get_num_tets() << " tetrahedra\n"
              << "ellipsoid ridge: " << surface.dual_vertex_count << " dual vertices, "
              << surface.vertices.size() - surface.dual_vertex_count << " polygon centers, "
              << surface.ridge_triangles.size() << " triangles\n"
              << "wrote ellipsoid_ridge.ply\n";
}
