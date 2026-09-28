#include "ellipsoid_example.hpp"

int main(int argc, char* argv[]) {
    // Start adaptively at 4x4x4; the original example remains uniform by default.
    return run_ellipsoid_example(argc, argv, true);
}
