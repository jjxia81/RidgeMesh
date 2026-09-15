#include <cassert>
#include "ridge_surface/ridge_surface.hpp"
int main() {
  using namespace ridge_surface;
  auto mesh=extract_height_ridges(ScalarField3D([](const Vec3& p) { return -p.x*p.x-.1*p.y*p.y-.05*p.z*p.z; }), {{-1,-1,-1},{1,1,1}}, SurfaceOptions{13,12,12});
  assert(!mesh.vertices.empty());
  assert(!mesh.ridge_triangles.empty());
}
