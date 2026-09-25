#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

#include <mtet/grid.h>

#include "ridge_surface/ridge_surface.hpp"

namespace {

struct OutputChoice {
    ridge_surface::PolygonTriangulation triangulation;
    const char* filename;
};

OutputChoice choose_output(int argc, char* argv[]) {
    using ridge_surface::PolygonTriangulation;
    if (argc == 1) {
        return {PolygonTriangulation::center_fan, "ellipsoid_ridge.ply"};
    }
    if (argc == 3 && std::string_view(argv[1]) == "--triangulation") {
        const std::string_view mode(argv[2]);
        if (mode == "center_fan") {
            return {PolygonTriangulation::center_fan, "ellipsoid_ridge.ply"};
        }
        if (mode == "vertex_fan") {
            return {PolygonTriangulation::vertex_fan, "ellipsoid_ridge_vertex_fan.ply"};
        }
        if (mode == "polygons_only") {
            return {PolygonTriangulation::polygons_only, "ellipsoid_ridge_polygons.ply"};
        }
    }
    throw std::invalid_argument(
        "usage: ellipsoid_example [--triangulation center_fan|vertex_fan|polygons_only]");
}

void write_ridge_ply(const ridge_surface::SurfaceMesh& mesh, const OutputChoice& choice) {
    const bool polygon_faces = choice.triangulation ==
        ridge_surface::PolygonTriangulation::polygons_only;
    const std::size_t face_count = polygon_faces
        ? mesh.ridge_polygons.size() : mesh.ridge_triangles.size();
    const char* filename = choice.filename;
    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("could not open ellipsoid PLY for writing");
    }

    output << "ply\nformat ascii 1.0\n"
           << "element vertex " << mesh.vertices.size() << "\n"
           << "property float x\nproperty float y\nproperty float z\n"
           << "element face " << face_count << "\n"
           << "property list uchar int vertex_indices\nend_header\n";
    for (const ridge_surface::Vec3& point : mesh.vertices) {
        output << point.x << ' ' << point.y << ' ' << point.z << '\n';
    }
    if (polygon_faces) {
        for (const ridge_surface::Polygon& polygon : mesh.ridge_polygons) {
            if (polygon.indices.size() > std::numeric_limits<unsigned char>::max()) {
                throw std::runtime_error("dual polygon exceeds the PLY face-size limit");
            }
            output << polygon.indices.size();
            for (const std::size_t index : polygon.indices) {
                output << ' ' << index;
            }
            output << '\n';
        }
    } else {
        for (const ridge_surface::Triangle& triangle : mesh.ridge_triangles) {
            output << "3 " << triangle.indices[0] << ' ' << triangle.indices[1] << ' '
                   << triangle.indices[2] << '\n';
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    using namespace ridge_surface;
    const OutputChoice choice = choose_output(argc, argv);

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
    options.polygon_triangulation = choice.triangulation;

    const SurfaceMesh surface = extract_height_ridges(field, grid, options);
    const bool polygon_faces = choice.triangulation == PolygonTriangulation::polygons_only;
    if (polygon_faces ? surface.ridge_polygons.empty() : surface.ridge_triangles.empty()) {
        throw std::runtime_error("ellipsoid example did not extract a ridge mesh");
    }
    write_ridge_ply(surface, choice);

    std::cout << "ellipsoid grid: " << grid.get_num_vertices() << " vertices, "
              << grid.get_num_tets() << " tetrahedra\n"
              << "ellipsoid ridge: " << surface.dual_vertex_count << " dual vertices, "
              << surface.vertices.size() - surface.dual_vertex_count << " polygon centers, "
              << (polygon_faces ? surface.ridge_polygons.size() : surface.ridge_triangles.size())
              << (polygon_faces ? " polygons\n" : " triangles\n")
              << "wrote " << choice.filename << '\n';
}
