#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ridge_surface/ridge_surface.hpp"

namespace py = pybind11;
using namespace ridge_surface;

namespace {

Vec3 as_vec3(py::handle object) {
    const py::sequence values = py::cast<py::sequence>(object);
    if (values.size() != 3) {
        throw py::value_error("expected a sequence of exactly 3 values");
    }
    return {values[0].cast<double>(), values[1].cast<double>(), values[2].cast<double>()};
}

Mat3 as_mat3(py::handle object) {
    const py::sequence rows = py::cast<py::sequence>(object);
    if (rows.size() != 3) {
        throw py::value_error("expected a 3 by 3 Hessian matrix");
    }

    Mat3 matrix{};
    for (int row = 0; row < 3; ++row) {
        const py::sequence values = py::cast<py::sequence>(rows[row]);
        if (values.size() != 3) {
            throw py::value_error("expected a 3 by 3 Hessian matrix");
        }
        for (int column = 0; column < 3; ++column) {
            matrix[row][column] = values[column].cast<double>();
        }
    }
    return matrix;
}

SurfaceMesh extract_from_scalar(
    const py::function& function,
    const Bounds3D& bounds,
    const SurfaceOptions& options) {
    ScalarField3D field = [function](const Vec3& point) {
        py::gil_scoped_acquire acquire;
        return function(point.x, point.y, point.z).cast<double>();
    };
    return extract_height_ridges(field, bounds, options);
}

SurfaceMesh extract_from_derivatives(
    const py::function& gradient,
    const py::function& hessian,
    const Bounds3D& bounds,
    const SurfaceOptions& options,
    const py::object& value) {
    DifferentialField3D field;
    if (!value.is_none()) {
        const py::function scalar_value = value.cast<py::function>();
        field.value = [scalar_value](const Vec3& point) {
            py::gil_scoped_acquire acquire;
            return scalar_value(point.x, point.y, point.z).cast<double>();
        };
    }
    field.gradient = [gradient](const Vec3& point) {
        py::gil_scoped_acquire acquire;
        return as_vec3(gradient(point.x, point.y, point.z));
    };
    field.hessian = [hessian](const Vec3& point) {
        py::gil_scoped_acquire acquire;
        return as_mat3(hessian(point.x, point.y, point.z));
    };
    return extract_height_ridges(field, bounds, options);
}

} // namespace

PYBIND11_MODULE(ridge_surface, module) {
    module.doc() = "Strong height-ridge and valley extraction on an MTet TET6 grid.";

    py::class_<Vec3>(module, "Vec3")
        .def(py::init<double, double, double>(), py::arg("x") = 0.0, py::arg("y") = 0.0, py::arg("z") = 0.0)
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z);

    py::class_<Bounds3D>(module, "Bounds3D")
        .def(py::init<Vec3, Vec3>(), py::arg("min"), py::arg("max"))
        .def_readwrite("min", &Bounds3D::min)
        .def_readwrite("max", &Bounds3D::max);

    py::enum_<RefinementTarget>(module, "RefinementTarget")
        .value("none", RefinementTarget::none)
        .value("ridges", RefinementTarget::ridges)
        .value("valleys", RefinementTarget::valleys)
        .value("ridges_and_valleys", RefinementTarget::ridges_and_valleys);

    py::enum_<PolygonTriangulation>(module, "PolygonTriangulation")
        .value("center_fan", PolygonTriangulation::center_fan)
        .value("vertex_fan", PolygonTriangulation::vertex_fan)
        .value("polygons_only", PolygonTriangulation::polygons_only);

    py::enum_<RefinementPipeline>(module, "RefinementPipeline")
        .value("curvature_orientation", RefinementPipeline::curvature_orientation)
        .value("legacy_crossing", RefinementPipeline::legacy_crossing);

    py::enum_<LinearitySampling>(module, "LinearitySampling")
        .value("cubic_16", LinearitySampling::cubic_16)
        .value("face_centers_5", LinearitySampling::face_centers_5)
        .value("disabled", LinearitySampling::disabled);

    py::enum_<RidgeCurvatureFilter>(module, "RidgeCurvatureFilter")
        .value("both_endpoints", RidgeCurvatureFilter::both_endpoints)
        .value("crossing_point", RidgeCurvatureFilter::crossing_point);

    py::class_<LongestEdgeRefinementOptions>(module, "LongestEdgeRefinementOptions")
        .def(py::init<>())
        .def_readwrite("target", &LongestEdgeRefinementOptions::target)
        .def_readwrite("max_splits", &LongestEdgeRefinementOptions::max_splits)
        .def_readwrite("minimum_edge_length", &LongestEdgeRefinementOptions::minimum_edge_length)
        .def_readwrite("pipeline", &LongestEdgeRefinementOptions::pipeline)
        .def_readwrite("curvature_zero_tolerance", &LongestEdgeRefinementOptions::curvature_zero_tolerance)
        .def_readwrite("refine_curvature_boundary", &LongestEdgeRefinementOptions::refine_curvature_boundary)
        .def_readwrite("refine_orientation_boundary", &LongestEdgeRefinementOptions::refine_orientation_boundary)
        .def_readwrite("linearity_sampling", &LongestEdgeRefinementOptions::linearity_sampling)
        .def_readwrite("linearity_absolute_tolerance", &LongestEdgeRefinementOptions::linearity_absolute_tolerance)
        .def_readwrite("linearity_relative_tolerance", &LongestEdgeRefinementOptions::linearity_relative_tolerance);

    py::class_<SurfaceOptions>(module, "SurfaceOptions")
        .def(py::init<>())
        .def_readwrite("nx", &SurfaceOptions::nx)
        .def_readwrite("ny", &SurfaceOptions::ny)
        .def_readwrite("nz", &SurfaceOptions::nz)
        .def_readwrite("longest_edge_refinement", &SurfaceOptions::longest_edge_refinement)
        .def_readwrite("surface_target", &SurfaceOptions::surface_target)
        .def_readwrite("subdivide_roots", &SurfaceOptions::subdivide_roots)
        .def_readwrite("root_iterations", &SurfaceOptions::root_iterations)
        .def_readwrite("root_tolerance", &SurfaceOptions::root_tolerance)
        .def_readwrite("minimum_curvature_sum", &SurfaceOptions::minimum_curvature_sum)
        .def_readwrite("ridge_curvature_filter", &SurfaceOptions::ridge_curvature_filter)
        .def_readwrite("check_crossing_point_curvature", &SurfaceOptions::check_crossing_point_curvature)
        .def_readwrite("minimum_ridge_field_value", &SurfaceOptions::minimum_ridge_field_value)
        .def_readwrite("finite_difference_step", &SurfaceOptions::finite_difference_step)
        .def_readwrite("retain_dual_polygons", &SurfaceOptions::retain_dual_polygons)
        .def_readwrite("polygon_triangulation", &SurfaceOptions::polygon_triangulation);

    py::class_<Triangle>(module, "Triangle")
        .def_readonly("indices", &Triangle::indices);

    py::class_<Polygon>(module, "Polygon")
        .def_readonly("indices", &Polygon::indices);

    py::class_<SurfaceMesh>(module, "SurfaceMesh")
        .def_readonly("vertices", &SurfaceMesh::vertices)
        .def_readonly("dual_vertex_count", &SurfaceMesh::dual_vertex_count)
        .def_readonly("ridge_triangles", &SurfaceMesh::ridge_triangles)
        .def_readonly("valley_triangles", &SurfaceMesh::valley_triangles)
        .def_readonly("ridge_polygons", &SurfaceMesh::ridge_polygons)
        .def_readonly("valley_polygons", &SurfaceMesh::valley_polygons);

    module.def(
        "extract_height_ridges",
        &extract_from_scalar,
        py::arg("function"),
        py::arg("bounds"),
        py::arg("options") = SurfaceOptions{},
        "Extract ridges/valleys from f(x, y, z); derivatives use finite differences.");
    for (const bool use_sixteen : {true, false}) {
        module.def(use_sixteen ? "directional_linearity_error_16" : "directional_linearity_error_5",
            [use_sixteen](const py::function& gradient, const py::function& hessian,
                          const std::array<Vec3, 4>& vertices, int direction_index) {
                DifferentialField3D field;
                field.gradient = [gradient](const Vec3& p) {
                    py::gil_scoped_acquire acquire;
                    return as_vec3(gradient(p.x, p.y, p.z));
                };
                field.hessian = [hessian](const Vec3& p) {
                    py::gil_scoped_acquire acquire;
                    return as_mat3(hessian(p.x, p.y, p.z));
                };
                return use_sixteen ? directional_linearity_error_16(field, vertices, direction_index)
                                   : directional_linearity_error_5(field, vertices, direction_index);
            }, py::arg("gradient"), py::arg("hessian"), py::arg("vertices"),
               py::arg("direction_index") = 0);
    }
    module.def(
        "extract_height_ridges_from_derivatives",
        &extract_from_derivatives,
        py::arg("gradient"),
        py::arg("hessian"),
        py::arg("bounds"),
        py::arg("options") = SurfaceOptions{},
        py::arg("value") = py::none(),
        "Extract ridges/valleys using analytic gradient(x,y,z) and hessian(x,y,z) callbacks.");
}
