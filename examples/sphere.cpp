#include <iostream>
#include <fstream>
#include "ridge_surface/ridge_surface.hpp"

int main() {
  using namespace ridge_surface;
  // f = -x^2 - .1y^2 - .05z^2: the strong ridge is the plane x = 0.
  DifferentialField3D field{
    [](const Vec3& p) { return Vec3{-2*p.x, -.2*p.y, -.1*p.z}; },
    [](const Vec3&) { return Mat3{{{{-2,0,0}},{{0,-.2,0}},{{0,0,-.1}}}}; }
  };
  // Use an odd x resolution: x=0 then falls between grid planes rather than
  // exactly on one, avoiding the inherent zero-at-vertex ambiguity.
  SurfaceOptions options; options.nx=25; options.ny=options.nz=24;
  auto mesh=extract_height_ridges(field, {{-1,-1,-1},{1,1,1}}, options);
  std::ofstream ply("ridge_example.ply");
  if (!ply) { std::cerr << "Could not open ridge_example.ply for writing\n"; return 1; }
  const std::size_t faces=mesh.ridge_triangles.size()+mesh.valley_triangles.size();
  ply << "ply\nformat ascii 1.0\n"
      << "element vertex " << mesh.vertices.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "element face " << faces << "\n"
      << "property list uchar int vertex_indices\nend_header\n";
  for (const Vec3& p : mesh.vertices) ply << p.x << ' ' << p.y << ' ' << p.z << '\n';
  const auto write_faces=[&](const std::vector<Triangle>& triangles) {
    for (const Triangle& t : triangles)
      ply << "3 " << t.indices[0] << ' ' << t.indices[1] << ' ' << t.indices[2] << '\n';
  };
  write_faces(mesh.ridge_triangles);
  write_faces(mesh.valley_triangles);
  std::cout << "dual vertices: " << mesh.vertices.size()
            << ", ridge triangles: " << mesh.ridge_triangles.size()
            << ", valley triangles: " << mesh.valley_triangles.size()
            << "; wrote ridge_example.ply\n";
}
