"""Matplotlib helpers for inspecting scalar fields and extracted ridge meshes."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

Axis = Literal["x", "y", "z"]


def plot_scalar_slice(
    function: Callable[[float, float, float], float],
    bounds,
    *,
    axis: Axis = "z",
    position: float = 0.0,
    resolution: int = 250,
    levels: int = 64,
    cmap: str = "viridis",
    ax=None,
):
    """Plot a two-dimensional slice of ``function`` with Matplotlib.

    ``axis`` selects the fixed coordinate and ``position`` gives its value.
    The returned ``(figure, axes)`` lets callers further customize or save it.
    """
    import matplotlib.pyplot as plt

    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y', or 'z'")

    coordinates = {
        "x": (bounds.min.x, bounds.max.x),
        "y": (bounds.min.y, bounds.max.y),
        "z": (bounds.min.z, bounds.max.z),
    }
    free_axes = [name for name in ("x", "y", "z") if name != axis]
    first = np.linspace(*coordinates[free_axes[0]], resolution)
    second = np.linspace(*coordinates[free_axes[1]], resolution)
    grid_first, grid_second = np.meshgrid(first, second, indexing="xy")

    values = np.empty_like(grid_first, dtype=float)
    for row in range(resolution):
        for column in range(resolution):
            point = {axis: position, free_axes[0]: grid_first[row, column], free_axes[1]: grid_second[row, column]}
            values[row, column] = function(point["x"], point["y"], point["z"])

    if ax is None:
        figure, ax = plt.subplots(constrained_layout=True)
    else:
        figure = ax.figure

    contour = ax.contourf(grid_first, grid_second, values, levels=levels, cmap=cmap)
    figure.colorbar(contour, ax=ax, label="f(x, y, z)")
    ax.contour(grid_first, grid_second, values, levels=[0.0], colors="white", linewidths=1.0)
    ax.set_xlabel(free_axes[0])
    ax.set_ylabel(free_axes[1])
    ax.set_title(f"{axis} = {position:g}")
    ax.set_aspect("equal")
    return figure, ax


def plot_mesh(mesh, *, show_ridges: bool = True, show_valleys: bool = True, ax=None):
    """Plot a ``ridge_surface.SurfaceMesh`` in a Matplotlib 3D axes."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        figure = plt.figure(constrained_layout=True)
        ax = figure.add_subplot(projection="3d")
    else:
        figure = ax.figure

    vertices = np.asarray([(vertex.x, vertex.y, vertex.z) for vertex in mesh.vertices], dtype=float)
    if vertices.size == 0:
        return figure, ax

    def add_triangles(triangles, color: str, label: str):
        if not triangles:
            return
        faces = [vertices[np.asarray(triangle.indices, dtype=int)] for triangle in triangles]
        collection = Poly3DCollection(faces, facecolor=color, edgecolor="none", alpha=0.8, label=label)
        ax.add_collection3d(collection)

    if show_ridges:
        add_triangles(mesh.ridge_triangles, "tab:red", "ridge")
    if show_valleys:
        add_triangles(mesh.valley_triangles, "tab:blue", "valley")

    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = max(float((maximum - minimum).max()) * 0.5, 1e-12)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return figure, ax
