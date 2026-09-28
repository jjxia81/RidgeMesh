#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <mtet/grid.h>

#include "ridge_surface/ridge_surface.hpp"
#include "ellipsoid_example.hpp"

namespace {

struct OutputChoice {
    ridge_surface::PolygonTriangulation triangulation =
        ridge_surface::PolygonTriangulation::center_fan;
    bool adaptive = false;
    ridge_surface::RefinementPipeline refinement_pipeline =
        ridge_surface::RefinementPipeline::curvature_orientation;
    std::string filename;
    std::string grid_filename;
    bool help = false;
    double absolute_tolerance = 1e-3;
    ridge_surface::LinearitySampling sampling = ridge_surface::LinearitySampling::cubic_16;
    int max_splits = 100'000;
    double minimum_edge_length = 0.005;
    ridge_surface::RidgeCurvatureFilter curvature_filter = ridge_surface::RidgeCurvatureFilter::both_endpoints;
};

double nonnegative_number(const std::string& text, const char* option) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(option) + " requires a finite nonnegative number");
    }
    return value;
}

void print_usage(bool linearity_example) {
    std::cout << "usage: " << (linearity_example ? "ellipsoid_linearity_example" : "ellipsoid_example")
              << " [--adaptive] [--le VALUE]\n"
              << "  [--linearity-sampling cubic_16|face_centers_5|disabled]\n"
              << "  [--max-splits INTEGER] [--minimum-edge-length VALUE]\n"
              << "  [--triangulation center_fan|vertex_fan|polygons_only]\n"
              << "  [--refinement-pipeline curvature_orientation|legacy_crossing]\n"
              << "  [--ridge-curvature-filter both_endpoints|crossing_point]\n"
              << "defaults: tolerance=0.001, sampling=cubic_16, max-splits=100000, "
                 "minimum-edge-length=0.005; Boundary II disabled\n";
}

OutputChoice choose_output(int argc, char* argv[], bool linearity_example) {
    using ridge_surface::PolygonTriangulation;
    OutputChoice choice;
    choice.adaptive = linearity_example;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            choice.help = true;
            return choice;
        } else if (argument == "--adaptive") {
            choice.adaptive = true;
        } else if ((argument == "--le" || argument == "--linearity-absolute-tolerance") && index + 1 < argc) {
            choice.absolute_tolerance = nonnegative_number(argv[++index], "--le");
        } else if (argument == "--minimum-edge-length" && index + 1 < argc) {
            choice.minimum_edge_length = nonnegative_number(argv[++index], "--minimum-edge-length");
        } else if (argument == "--max-splits" && index + 1 < argc) {
            const std::string text(argv[++index]);
            std::size_t consumed = 0;
            choice.max_splits = std::stoi(text, &consumed);
            if (consumed != text.size() || choice.max_splits < 0) {
                throw std::invalid_argument("--max-splits requires a nonnegative integer");
            }
        } else if (argument == "--linearity-sampling" && index + 1 < argc) {
            const std::string_view mode(argv[++index]);
            if (mode == "cubic_16") {
                choice.sampling = ridge_surface::LinearitySampling::cubic_16;
            } else if (mode == "face_centers_5") {
                choice.sampling = ridge_surface::LinearitySampling::face_centers_5;
            } else if (mode == "disabled") {
                choice.sampling = ridge_surface::LinearitySampling::disabled;
            } else {
                throw std::invalid_argument("unknown linearity sampling mode");
            }
        } else if (argument == "--ridge-curvature-filter" && index + 1 < argc) {
            const std::string_view mode(argv[++index]);
            if (mode == "both_endpoints") {
                choice.curvature_filter = ridge_surface::RidgeCurvatureFilter::both_endpoints;
            } else if (mode == "crossing_point") {
                choice.curvature_filter = ridge_surface::RidgeCurvatureFilter::crossing_point;
            } else {
                throw std::invalid_argument("unknown ridge curvature filter");
            }
        } else if (argument == "--refinement-pipeline" && index + 1 < argc) {
            const std::string_view mode(argv[++index]);
            if (mode == "curvature_orientation") {
                choice.refinement_pipeline = ridge_surface::RefinementPipeline::curvature_orientation;
            } else if (mode == "legacy_crossing") {
                choice.refinement_pipeline = ridge_surface::RefinementPipeline::legacy_crossing;
            } else {
                throw std::invalid_argument("unknown refinement pipeline");
            }
        } else if (argument == "--triangulation" && index + 1 < argc) {
            const std::string_view mode(argv[++index]);
            if (mode == "center_fan") {
                choice.triangulation = PolygonTriangulation::center_fan;
            } else if (mode == "vertex_fan") {
                choice.triangulation = PolygonTriangulation::vertex_fan;
            } else if (mode == "polygons_only") {
                choice.triangulation = PolygonTriangulation::polygons_only;
            } else {
                throw std::invalid_argument("unknown triangulation mode");
            }
        } else {
            throw std::invalid_argument("unknown or incomplete argument: " + std::string(argument));
        }
    }

    choice.filename = linearity_example ? "ellipsoid_linearity_ridge"
        : (choice.adaptive ? "ellipsoid_adaptive_ridge" : "ellipsoid_ridge");
    choice.grid_filename = linearity_example ? "ellipsoid_linearity_grid_wireframe.ply"
                                            : "ellipsoid_adaptive_grid_wireframe.ply";
    if (choice.curvature_filter == ridge_surface::RidgeCurvatureFilter::crossing_point) {
        choice.filename += "_crossing_point";
    }
    if (choice.triangulation == PolygonTriangulation::vertex_fan) {
        choice.filename += "_vertex_fan";
    } else if (choice.triangulation == PolygonTriangulation::polygons_only) {
        choice.filename += "_polygons";
    }
    choice.filename += ".ply";
    return choice;
}

void write_ridge_ply(const ridge_surface::SurfaceMesh& mesh, const OutputChoice& choice) {
    const bool polygon_faces = choice.triangulation ==
        ridge_surface::PolygonTriangulation::polygons_only;
    const std::size_t face_count = polygon_faces
        ? mesh.ridge_polygons.size() : mesh.ridge_triangles.size();
    std::ofstream output(choice.filename);
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

// Show the final MTet edges so the adaptive region can be inspected separately.
void write_grid_wireframe_ply(mtet::MTetMesh& grid, const char* filename) {
    std::map<std::uint64_t, std::size_t> output_vertex_by_id;
    std::vector<std::array<double, 3>> vertices;
    grid.seq_foreach_vertex([&](mtet::VertexId vertex_id, auto point) {
        output_vertex_by_id.emplace(vertex_id.value_of(), vertices.size());
        vertices.push_back({point[0], point[1], point[2]});
    });

    std::set<std::pair<std::size_t, std::size_t>> edges;
    grid.seq_foreach_tet([&](mtet::TetId tet_id, auto) {
        grid.foreach_edge_in_tet(tet_id, [&](mtet::EdgeId, mtet::VertexId first,
                                              mtet::VertexId second) {
            const std::size_t a = output_vertex_by_id.at(first.value_of());
            const std::size_t b = output_vertex_by_id.at(second.value_of());
            edges.emplace(std::min(a, b), std::max(a, b));
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
    for (const auto& [a, b] : edges) {
        output << a << ' ' << b << '\n';
    }
}

} // namespace

int run_ellipsoid_example(int argc, char* argv[], bool linearity_example) try {
    using namespace ridge_surface;
    const OutputChoice choice = choose_output(argc, argv, linearity_example);
    if (choice.help) {
        print_usage(linearity_example);
        return 0;
    }

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
    const std::array<std::size_t, 3> resolution = choice.adaptive
        ? std::array<std::size_t, 3>{4, 4, 4}
        : std::array<std::size_t, 3>{64, 64, 64};
    mtet::MTetMesh grid = mtet::generate_tet_grid(
        resolution, {-0.98, -1.0, -0.40}, {1.02, 1.0, 0.40}, mtet::TET6);
    const std::size_t initial_vertex_count = grid.get_num_vertices();
    const std::size_t initial_tet_count = grid.get_num_tets();
    SurfaceOptions options;
    options.surface_target = RefinementTarget::ridges;
    options.ridge_curvature_filter = choice.curvature_filter;
    options.polygon_triangulation = choice.triangulation;
    options.longest_edge_refinement.pipeline = choice.refinement_pipeline;
    options.longest_edge_refinement.linearity_absolute_tolerance = choice.absolute_tolerance;
    options.longest_edge_refinement.linearity_relative_tolerance = 0.0;
    options.longest_edge_refinement.linearity_sampling = choice.sampling;
    if (choice.adaptive) {
        options.longest_edge_refinement.target = RefinementTarget::ridges;
        options.longest_edge_refinement.max_splits = choice.max_splits;
        options.longest_edge_refinement.minimum_edge_length = choice.minimum_edge_length;
    }

    const SurfaceMesh surface = extract_height_ridges(field, grid, options);
    const bool polygon_faces = choice.triangulation == PolygonTriangulation::polygons_only;
    if (polygon_faces ? surface.ridge_polygons.empty() : surface.ridge_triangles.empty()) {
        throw std::runtime_error("ellipsoid example did not extract a ridge mesh");
    }
    write_ridge_ply(surface, choice);
    if (choice.adaptive) {
        write_grid_wireframe_ply(grid, choice.grid_filename.c_str());
    }

    std::cout << "ellipsoid initial grid: " << initial_vertex_count << " vertices, "
              << initial_tet_count << " tetrahedra\n"
              << "ellipsoid final grid: " << grid.get_num_vertices() << " vertices, "
              << grid.get_num_tets() << " tetrahedra\n"
              << "ellipsoid ridge: " << surface.dual_vertex_count << " dual vertices, "
              << surface.vertices.size() - surface.dual_vertex_count << " polygon centers, "
              << (polygon_faces ? surface.ridge_polygons.size() : surface.ridge_triangles.size())
              << (polygon_faces ? " polygons\n" : " triangles\n")
              << "wrote " << choice.filename << '\n';
    if (choice.adaptive) {
        std::cout << "actual edge splits: " << grid.get_num_vertices() - initial_vertex_count
                  << " / " << choice.max_splits << "\n"
                  << "linearity absolute tolerance: " << choice.absolute_tolerance
                  << ", relative tolerance: 0, minimum edge length: " << choice.minimum_edge_length
                  << "\nwrote " << choice.grid_filename << '\n';
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    print_usage(linearity_example);
    return 1;
}

#ifndef RIDGE_ELLIPSOID_NO_MAIN
int main(int argc, char* argv[]) {
    return run_ellipsoid_example(argc, argv, false);
}
#endif
