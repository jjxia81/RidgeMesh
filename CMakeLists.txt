cmake_minimum_required(VERSION 3.16)
project(ridge_surface VERSION 0.1.0 LANGUAGES CXX)

add_subdirectory(third_party/mtet EXCLUDE_FROM_ALL)
add_subdirectory(third_party/eigen EXCLUDE_FROM_ALL)

option(RIDGE_SURFACE_BUILD_PYTHON "Build the native Python extension" ON)
if(RIDGE_SURFACE_BUILD_PYTHON)
  add_subdirectory(third_party/pybind11 EXCLUDE_FROM_ALL)
endif()
add_library(ridge_surface src/ridge_surface.cpp)
target_include_directories(ridge_surface PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(ridge_surface PUBLIC cxx_std_20)
# MTet is public because the API accepts an mtet::MTetMesh supplied by callers.
target_link_libraries(ridge_surface PUBLIC mtet::mtet PRIVATE Eigen3::Eigen)
if(MSVC)
  # MTet's fmt dependency requires Unicode source support on MSVC.
  target_compile_options(mtet PUBLIC /utf-8)
  target_compile_options(ridge_surface PRIVATE /utf-8)
endif()
target_compile_options(ridge_surface PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>)

add_executable(ridge_example examples/sphere.cpp)
target_link_libraries(ridge_example PRIVATE ridge_surface)

enable_testing()
add_executable(ridge_test tests/smoke_test.cpp)
target_link_libraries(ridge_test PRIVATE ridge_surface)
add_test(NAME ridge_smoke COMMAND ridge_test)

if(RIDGE_SURFACE_BUILD_PYTHON)
  pybind11_add_module(ridge_surface_python python/bindings.cpp)
  target_link_libraries(ridge_surface_python PRIVATE ridge_surface)
  set_target_properties(ridge_surface_python PROPERTIES
    OUTPUT_NAME ridge_surface
    ARCHIVE_OUTPUT_NAME ridge_surface_python)
  if(MSVC)
    target_compile_options(ridge_surface_python PRIVATE /utf-8)
  endif()
  add_custom_command(TARGET ridge_surface_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_SOURCE_DIR}/python/ridge_surface_plot.py
      $<TARGET_FILE_DIR:ridge_surface_python>)
endif()

if(WIN32)
  set(ridge_runtime_targets ridge_example ridge_test)
  if(RIDGE_SURFACE_BUILD_PYTHON)
    list(APPEND ridge_runtime_targets ridge_surface_python)
  endif()
  foreach(target IN LISTS ridge_runtime_targets)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:nanothread> $<TARGET_FILE_DIR:${target}>)
  endforeach()
endif()
