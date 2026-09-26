"""PyTorch and GeoUDF adapters for the RidgeMesh native surfacer.

The adapter accepts scalar-field gradients from a model output, autograd, or
central differences. A GeoUDF caller must transform its learned UDF gradient
when it transforms the UDF into a different scalar field.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ridge_surface import Bounds3D, SurfaceOptions, Vec3, extract_height_ridges_from_derivatives


def _move_tensors(value: Any, device):
    """Recursively move a GeoUDF input dictionary to ``device``."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    return value


class TorchFieldAdapter:
    """Expose a differentiable PyTorch UDF as RidgeMesh derivative callbacks.

    ``model(context, query)`` must return a scalar UDF tensor, a tuple whose
    first item is the UDF, or a mapping containing ``"udf"``. For
    ``gradient_mode="network"``, the second tuple item (or ``"gradient"``
    mapping entry) must approximate the gradient of that same scalar field.
    GeoUDF's native
    convention is a query tensor with shape ``(B, 3, M)``; this adapter uses
    ``(1, 3, 1)`` for each requested point.

    Set ``gradient_mode="finite_difference"`` to estimate only the gradient
    from central scalar-UDF differences while retaining the chosen Hessian
    backend. Set ``derivative_mode="finite_difference"`` to estimate the
    Hessian from scalar differences. With a network gradient, that mode
    instead differences the network gradient for the Hessian. Derivatives
    are cached by point
    because the C++ surfacer asks for a Hessian and gradient at the same
    locations. Set ``uniform_grid_batch_size`` in ``extract_torch_udf`` to
    precompute either derivative mode in CUDA batches on a uniform base grid.
    Adaptive midpoint samples remain pointwise.
    """

    def __init__(
        self,
        model: Callable,
        context: Optional[Mapping[str, Any]] = None,
        *,
        device: Optional[str] = None,
        dtype=None,
        move_context: bool = True,
        derivative_mode: str = "autograd",
        gradient_mode: Optional[str] = None,
        hessian_backend: str = "functional",
        finite_difference_step: float = 5e-3,
    ):
        import torch

        self.model = model
        self.device = torch.device(device) if device is not None else self._model_device(model)
        self.dtype = dtype if dtype is not None else torch.float32
        self.context = _move_tensors(context or {}, self.device) if move_context else (context or {})
        if derivative_mode not in ("autograd", "finite_difference"):
            raise ValueError("derivative_mode must be 'autograd' or 'finite_difference'")
        if gradient_mode is None:
            gradient_mode = derivative_mode
        if gradient_mode not in ("autograd", "finite_difference", "network"):
            raise ValueError("gradient_mode must be 'autograd', 'finite_difference', or 'network'")
        if hessian_backend not in ("grad", "functional", "gradient_difference"):
            raise ValueError(
                "hessian_backend must be 'grad', 'functional', or 'gradient_difference'"
            )
        if finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive")
        if derivative_mode == "finite_difference" and gradient_mode == "network":
            # For a learned gradient, form the numerical Hessian from that
            # gradient. Reuse the existing gradient-difference branch, which
            # does not invoke autograd when gradient_mode is "network".
            derivative_mode = "autograd"
            hessian_backend = "gradient_difference"
        self.derivative_mode = derivative_mode
        self.gradient_mode = gradient_mode
        self.hessian_backend = hessian_backend
        self.finite_difference_step = float(finite_difference_step)
        self._cache: Dict[
            Tuple[float, float, float],
            Tuple[Tuple[float, float, float], List[List[float]]],
        ] = {}
        self._uniform_grid_samples = None

        if hasattr(model, "eval"):
            model.eval()

    @staticmethod
    def _model_device(model):
        import torch

        try:
            return next(model.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    @staticmethod
    def _udf_from_output(output):
        if isinstance(output, Mapping):
            if "udf" not in output:
                raise KeyError("model mapping output must contain a 'udf' tensor")
            return output["udf"]
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _finite_difference_gradients_batch(self, points):
        """Central differences of the scalar field at an ``(N, 3)`` point batch."""
        import torch

        h = self.finite_difference_step
        columns = []
        with torch.no_grad():
            for axis in range(3):
                direction = torch.zeros(3, device=self.device, dtype=self.dtype)
                direction[axis] = h
                positive = self._udf_from_output(
                    self.model(self.context, (points + direction).transpose(0, 1).unsqueeze(0))
                ).reshape(-1)
                negative = self._udf_from_output(
                    self.model(self.context, (points - direction).transpose(0, 1).unsqueeze(0))
                ).reshape(-1)
                if positive.numel() != points.shape[0] or negative.numel() != points.shape[0]:
                    raise ValueError("the model must return one scalar UDF value per query")
                columns.append((positive - negative) / (2.0 * h))
        return torch.stack(columns, dim=1)

    def _network_gradients_batch(self, points):
        """Read approximate gradients of the selected scalar field from the model."""
        import torch

        with torch.no_grad():
            output = self.model(self.context, points.transpose(0, 1).unsqueeze(0))
        if isinstance(output, Mapping):
            gradient = output.get("gradient")
        elif isinstance(output, (tuple, list)) and len(output) > 1:
            gradient = output[1]
        else:
            gradient = None
        if gradient is None:
            raise ValueError("network gradient mode requires a model gradient output")
        if gradient.shape == (1, points.shape[0], 3):
            gradient = gradient[0]
        if gradient.shape != (points.shape[0], 3):
            raise ValueError("network gradients must have shape (1, N, 3) or (N, 3)")
        if not torch.isfinite(gradient).all():
            raise ValueError("network gradients contain non-finite values")
        return gradient

    def _network_gradient_stencil(self, points, offsets):
        """Query center and offset network gradients in one model call."""
        stencil_points = (points.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1, 3)
        gradients = self._network_gradients_batch(stencil_points)
        return gradients.reshape(len(offsets), len(points), 3)

    def _network_gradient_difference_derivatives(self, point_coordinates):
        """Numerically differentiate the network gradient at an adaptive point."""
        import torch

        point = self._point_tensor(point_coordinates)
        h = self.finite_difference_step
        offsets = [torch.zeros(3, device=self.device, dtype=self.dtype)]
        for axis in range(3):
            direction = torch.zeros(3, device=self.device, dtype=self.dtype)
            direction[axis] = h
            offsets.extend((direction, -direction))

        all_gradients = self._network_gradient_stencil(point, torch.stack(offsets))
        gradient = all_gradients[0, 0]
        hessian = torch.stack([
            (all_gradients[1 + 2 * axis, 0] - all_gradients[2 + 2 * axis, 0]) / (2.0 * h)
            for axis in range(3)
        ], dim=1)
        hessian = 0.5 * (hessian + hessian.transpose(0, 1))
        return (
            tuple(float(value) for value in gradient.cpu()),
            [[float(value) for value in row] for row in hessian.cpu()],
        )

    def _autograd_gradients_batch(self, points):
        """Differentiate the scalar field once for each point in a batch."""
        import torch

        input_points = points.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            values = self._udf_from_output(
                self.model(self.context, input_points.transpose(0, 1).unsqueeze(0))
            ).reshape(-1)
            if values.numel() != input_points.shape[0]:
                raise ValueError("the model must return one scalar UDF value per query")
            gradients = torch.autograd.grad(values.sum(), input_points)[0]
        return gradients.detach()

    def _evaluate_derivatives(self, x: float, y: float, z: float):
        key = (float(x), float(y), float(z))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        uniform_sample = self._uniform_grid_sample(key)
        if uniform_sample is not None:
            return uniform_sample

        if self.gradient_mode == "network" and self.hessian_backend == "gradient_difference":
            result = self._network_gradient_difference_derivatives(key)
        elif self.derivative_mode == "finite_difference":
            result = self._evaluate_finite_difference_derivatives(key)
            if self.gradient_mode == "network":
                point = self._point_tensor(key)
                result = (
                    tuple(float(value) for value in self._network_gradients_batch(point)[0].cpu()),
                    result[1],
                )
            elif self.gradient_mode == "autograd":
                point = self._point_tensor(key)
                result = (
                    tuple(float(value) for value in self._autograd_gradients_batch(point)[0].cpu()),
                    result[1],
                )
        else:
            result = self._evaluate_autograd_derivatives(key)
        self._cache[key] = result
        return result

    def _point_tensor(self, coordinates):
        import torch

        return torch.tensor(
            coordinates, device=self.device, dtype=self.dtype
        ).reshape(1, 3)

    def precompute_uniform_grid(self, bounds, nx: int, ny: int, nz: int, batch_size: int) -> None:
        """Batch derivatives at all vertices of a uniform MTet grid.

        This removes the expensive C++ -> Python -> CUDA round trip for every
        initial-grid vertex. It assumes the model evaluates each query point
        independently, as GeoUDF does in evaluation mode. Points introduced by
        optional adaptive splitting are evaluated by the existing pointwise
        fallback path.
        """
        import numpy as np
        import torch

        if min(nx, ny, nz) < 1:
            raise ValueError("uniform derivative precomputation requires nx, ny, nz >= 1")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        lower = np.array((bounds.min.x, bounds.min.y, bounds.min.z), dtype=np.float64)
        upper = np.array((bounds.max.x, bounds.max.y, bounds.max.z), dtype=np.float64)
        # MTet's resolution is a cell count, so it creates resolution + 1
        # vertices along every axis.
        shape = (int(nx) + 1, int(ny) + 1, int(nz) + 1)
        axes = [
            torch.linspace(float(lower[axis]), float(upper[axis]), shape[axis],
                           device=self.device, dtype=self.dtype)
            for axis in range(3)
        ]
        mesh = torch.meshgrid(*axes, indexing="ij")
        points = torch.stack(mesh, dim=-1).reshape(-1, 3)
        gradients = np.empty((points.shape[0], 3), dtype=np.float32)
        hessians = np.empty((points.shape[0], 3, 3), dtype=np.float32)
        if self.derivative_mode == "autograd" and self.hessian_backend != "gradient_difference":
            print(
                "RidgeMesh: batching autograd derivatives for {} uniform-grid vertices "
                "({} queries per CUDA batch).".format(points.shape[0], batch_size),
                flush=True,
            )
            for first in range(0, points.shape[0], batch_size):
                last = min(first + batch_size, points.shape[0])
                batch = points[first:last].detach().clone().requires_grad_(True)
                query = batch.transpose(0, 1).unsqueeze(0)
                with torch.enable_grad():
                    values = self._udf_from_output(self.model(self.context, query)).reshape(-1)
                    if values.numel() != batch.shape[0]:
                        raise ValueError(
                            "the model must return one scalar UDF value for every batched query"
                        )
                    gradient = torch.autograd.grad(
                        values, batch, grad_outputs=torch.ones_like(values), create_graph=True
                    )[0]
                    rows = [
                        torch.autograd.grad(gradient[:, axis].sum(), batch, retain_graph=axis < 2)[0]
                        for axis in range(3)
                    ]
                    hessian = torch.stack(rows, dim=1)

                if self.gradient_mode == "finite_difference":
                    gradient = self._finite_difference_gradients_batch(batch.detach())
                elif self.gradient_mode == "network":
                    gradient = self._network_gradients_batch(batch.detach())
                gradients[first:last] = gradient.detach().cpu().numpy()
                hessians[first:last] = hessian.detach().cpu().numpy()
        elif self.derivative_mode == "finite_difference":
            # A 3-D central-difference Hessian needs 19 field samples per
            # vertex: the center, six axis neighbours, and twelve mixed-term
            # corners. Evaluate the whole 19-point stencil for every base
            # batch at once; the gradient can use a separately selected source.
            h = self.finite_difference_step
            offsets = [torch.zeros(3, device=self.device, dtype=self.dtype)]
            axis_indices = []
            for axis in range(3):
                direction = torch.zeros(3, device=self.device, dtype=self.dtype)
                direction[axis] = h
                axis_indices.append((len(offsets), len(offsets) + 1))
                offsets.extend((direction, -direction))
            mixed_indices = {}
            for first_axis in range(3):
                for second_axis in range(first_axis + 1, 3):
                    first_direction = torch.zeros(3, device=self.device, dtype=self.dtype)
                    second_direction = torch.zeros(3, device=self.device, dtype=self.dtype)
                    first_direction[first_axis] = h
                    second_direction[second_axis] = h
                    mixed_indices[(first_axis, second_axis)] = tuple(
                        range(len(offsets), len(offsets) + 4)
                    )
                    offsets.extend((
                        first_direction + second_direction,
                        first_direction - second_direction,
                        -first_direction + second_direction,
                        -first_direction - second_direction,
                    ))
            offsets = torch.stack(offsets)
            print(
                "RidgeMesh: batching finite-difference Hessians for {} uniform-grid vertices "
                "({} base vertices and {} stencil queries per CUDA batch).".format(
                    points.shape[0], batch_size, len(offsets) * batch_size
                ),
                flush=True,
            )
            for first in range(0, points.shape[0], batch_size):
                last = min(first + batch_size, points.shape[0])
                batch = points[first:last]
                stencil_points = (batch.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1, 3)
                query = stencil_points.transpose(0, 1).unsqueeze(0)
                with torch.no_grad():
                    values = self._udf_from_output(self.model(self.context, query)).reshape(
                        len(offsets), last - first
                    )
                center = values[0]
                gradient = torch.empty((last - first, 3), device=self.device, dtype=self.dtype)
                hessian = torch.zeros((last - first, 3, 3), device=self.device, dtype=self.dtype)
                for axis, (positive_index, negative_index) in enumerate(axis_indices):
                    positive, negative = values[positive_index], values[negative_index]
                    gradient[:, axis] = (positive - negative) / (2.0 * h)
                    hessian[:, axis, axis] = (positive - 2.0 * center + negative) / (h * h)
                for (first_axis, second_axis), indices in mixed_indices.items():
                    positive_positive, positive_negative, negative_positive, negative_negative = (
                        values[index] for index in indices
                    )
                    mixed = (
                        positive_positive - positive_negative - negative_positive + negative_negative
                    ) / (4.0 * h * h)
                    hessian[:, first_axis, second_axis] = mixed
                    hessian[:, second_axis, first_axis] = mixed
                if self.gradient_mode == "network":
                    gradient = self._network_gradients_batch(batch)
                elif self.gradient_mode == "autograd":
                    gradient = self._autograd_gradients_batch(batch)
                gradients[first:last] = gradient.cpu().numpy()
                hessians[first:last] = hessian.cpu().numpy()
        else:
            # Differentiate the scalar field once at the center and at its six
            # axis neighbours.  The Hessian column d(grad F)/d x_j comes from
            # the corresponding pair, then is symmetrized to suppress neural
            # evaluation noise.
            h = self.finite_difference_step
            offsets = [torch.zeros(3, device=self.device, dtype=self.dtype)]
            for axis in range(3):
                direction = torch.zeros(3, device=self.device, dtype=self.dtype)
                direction[axis] = h
                offsets.extend((direction, -direction))
            offsets = torch.stack(offsets)
            print(
                "RidgeMesh: batching gradient-difference Hessians for {} uniform-grid vertices "
                "({} base vertices and {} gradient queries per CUDA batch).".format(
                    points.shape[0], batch_size, len(offsets) * batch_size
                ),
                flush=True,
            )
            for first in range(0, points.shape[0], batch_size):
                last = min(first + batch_size, points.shape[0])
                batch = points[first:last]
                if self.gradient_mode == "network":
                    all_gradients = self._network_gradient_stencil(batch, offsets)
                else:
                    stencil_points = (batch.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1, 3)
                    stencil_points.requires_grad_(True)
                    query = stencil_points.transpose(0, 1).unsqueeze(0)
                    with torch.enable_grad():
                        values = self._udf_from_output(self.model(self.context, query)).reshape(-1)
                        if values.numel() != stencil_points.shape[0]:
                            raise ValueError(
                                "the model must return one scalar UDF value for every batched query"
                            )
                        all_gradients = torch.autograd.grad(
                            values, stencil_points, grad_outputs=torch.ones_like(values)
                        )[0].reshape(len(offsets), last - first, 3)
                gradient = all_gradients[0]
                hessian = torch.empty((last - first, 3, 3), device=self.device, dtype=self.dtype)
                for axis in range(3):
                    positive = all_gradients[1 + 2 * axis]
                    negative = all_gradients[2 + 2 * axis]
                    hessian[:, :, axis] = (positive - negative) / (2.0 * h)
                hessian = 0.5 * (hessian + hessian.transpose(1, 2))
                if self.gradient_mode == "finite_difference":
                    gradient = self._finite_difference_gradients_batch(batch)
                gradients[first:last] = gradient.detach().cpu().numpy()
                hessians[first:last] = hessian.detach().cpu().numpy()

        self._uniform_grid_samples = {
            "lower": lower,
            "upper": upper,
            "shape": shape,
            "step": (upper - lower) / (np.asarray(shape, dtype=np.float64) - 1.0),
            "gradient": gradients,
            "hessian": hessians,
        }

    def _uniform_grid_sample(self, point_coordinates):
        """Return a cached uniform-grid sample, or None for an adaptive point."""
        if self._uniform_grid_samples is None:
            return None

        import numpy as np

        cache = self._uniform_grid_samples
        point = np.asarray(point_coordinates, dtype=np.float64)
        coordinate = (point - cache["lower"]) / cache["step"]
        index = np.rint(coordinate).astype(np.int64)
        tolerance = 2e-5
        if (
            np.any(index < 0)
            or np.any(index >= np.asarray(cache["shape"]))
            or np.any(np.abs(coordinate - index) > tolerance)
        ):
            return None

        flat_index = np.ravel_multi_index(tuple(index), cache["shape"])
        gradient = tuple(float(value) for value in cache["gradient"][flat_index])
        hessian = [[float(value) for value in row] for row in cache["hessian"][flat_index]]
        return gradient, hessian

    def _evaluate_autograd_derivatives(self, point_coordinates):
        import torch

        point = torch.tensor(point_coordinates, device=self.device, dtype=self.dtype, requires_grad=True)
        query = point.reshape(1, 3, 1)
        with torch.enable_grad():
            udf = self._udf_from_output(self.model(self.context, query))
            if udf.numel() != 1:
                raise ValueError("the model must return one scalar UDF value for a (1, 3, 1) query")
            gradient = torch.autograd.grad(
                udf.reshape(()),
                point,
                create_graph=self.hessian_backend != "gradient_difference",
            )[0]
            if self.hessian_backend == "functional":
                def scalar_from_point(input_point):
                    input_query = input_point.reshape(1, 3, 1)
                    value = self._udf_from_output(self.model(self.context, input_query))
                    if value.numel() != 1:
                        raise ValueError(
                            "the model must return one scalar UDF value for a (1, 3, 1) query"
                        )
                    return value.reshape(())

                hessian = torch.autograd.functional.hessian(
                    scalar_from_point, point, create_graph=False, vectorize=False
                )
                hessian_rows = [hessian[index] for index in range(3)]
            elif self.hessian_backend == "grad":
                hessian_rows = [
                    torch.autograd.grad(gradient[index], point, retain_graph=index < 2)[0]
                    for index in range(3)
                ]
            else:
                h = self.finite_difference_step

                def gradient_at(input_point):
                    input_point = input_point.detach().clone().requires_grad_(True)
                    input_query = input_point.reshape(1, 3, 1)
                    value = self._udf_from_output(self.model(self.context, input_query))
                    if value.numel() != 1:
                        raise ValueError(
                            "the model must return one scalar UDF value for a (1, 3, 1) query"
                        )
                    return torch.autograd.grad(value.reshape(()), input_point)[0]

                columns = []
                for axis in range(3):
                    positive = point.detach().clone()
                    negative = point.detach().clone()
                    positive[axis] += h
                    negative[axis] -= h
                    positive_gradient = gradient_at(positive)
                    negative_gradient = gradient_at(negative)
                    columns.append((positive_gradient - negative_gradient) / (2.0 * h))
                hessian = torch.stack(columns, dim=1)
                hessian = 0.5 * (hessian + hessian.transpose(0, 1))
                hessian_rows = [hessian[index] for index in range(3)]

        if self.gradient_mode == "finite_difference":
            gradient = self._finite_difference_gradients_batch(point.detach().reshape(1, 3))[0]
        elif self.gradient_mode == "network":
            gradient = self._network_gradients_batch(point.detach().reshape(1, 3))[0]
        gradient_value = tuple(float(value) for value in gradient.detach().cpu())
        hessian_value = [
            [float(value) for value in row.detach().cpu()]
            for row in hessian_rows
        ]
        return gradient_value, hessian_value

    def _scalar_value(self, point_coordinates):
        """Evaluate one scalar UDF value without retaining an autograd graph."""
        import torch

        point = torch.tensor(point_coordinates, device=self.device, dtype=self.dtype)
        query = point.reshape(1, 3, 1)
        with torch.no_grad():
            udf = self._udf_from_output(self.model(self.context, query))
        if udf.numel() != 1:
            raise ValueError("the model must return one scalar UDF value for a (1, 3, 1) query")
        return float(udf.reshape(()).detach().cpu())

    def _evaluate_finite_difference_derivatives(self, point_coordinates):
        """Estimate gradient/Hessian from central scalar-UDF differences.

        The diagonal terms use second differences and off-diagonal terms use
        the symmetric four-corner stencil. This requires 19 scalar UDF calls
        per uncached point, so it is intended for correctness checks and
        modest adaptive grids rather than a dense neural-UDF sweep.
        """
        h = self.finite_difference_step
        center_value = self._scalar_value(point_coordinates)
        gradient = [0.0, 0.0, 0.0]
        hessian = [[0.0, 0.0, 0.0] for _ in range(3)]

        def shifted(first_axis, first_offset, second_axis=None, second_offset=0.0):
            point = list(point_coordinates)
            point[first_axis] += first_offset
            if second_axis is not None:
                point[second_axis] += second_offset
            return self._scalar_value(point)

        for axis in range(3):
            positive = shifted(axis, h)
            negative = shifted(axis, -h)
            gradient[axis] = (positive - negative) / (2.0 * h)
            hessian[axis][axis] = (positive - 2.0 * center_value + negative) / (h * h)

        for row in range(3):
            for column in range(row + 1, 3):
                mixed = (
                    shifted(row, h, column, h)
                    - shifted(row, h, column, -h)
                    - shifted(row, -h, column, h)
                    + shifted(row, -h, column, -h)
                ) / (4.0 * h * h)
                hessian[row][column] = mixed
                hessian[column][row] = mixed

        return tuple(gradient), hessian

    def gradient(self, x: float, y: float, z: float):
        """Return the selected scalar-field gradient in the format expected by pybind11."""
        return self._evaluate_derivatives(x, y, z)[0]

    def hessian(self, x: float, y: float, z: float):
        """Return the selected scalar-field Hessian in the format expected by pybind11."""
        return self._evaluate_derivatives(x, y, z)[1]

    def value(self, x: float, y: float, z: float):
        """Return the scalar field value for an optional crossing-value gate."""
        return self._scalar_value((x, y, z))


def extract_torch_udf(
    model: Callable,
    context: Optional[Mapping[str, Any]],
    bounds: Bounds3D,
    options: Optional[SurfaceOptions] = None,
    *,
    device: Optional[str] = None,
    dtype=None,
    derivative_mode: str = "autograd",
    gradient_mode: Optional[str] = None,
    hessian_backend: str = "functional",
    finite_difference_step: float = 5e-3,
    uniform_grid_batch_size: int = 0,
):
    """Extract a ridge/valley mesh from a PyTorch UDF such as GeoUDF.

    The model is called as ``model(context, query)`` with a GeoUDF-compatible
    query tensor of shape ``(1, 3, 1)``. Set ``derivative_mode`` to
    ``"finite_difference"`` to avoid second-order autograd. With a network
    gradient, this differences the network gradient rather than scalar values.
    Set
    ``gradient_mode="finite_difference"`` to replace only the gradient with
    central differences of the same scalar field. Use ``"network"`` when the
    model returns a matching approximate gradient as its second output. With autograd,
    ``hessian_backend="functional"`` uses ``torch.autograd.functional.hessian``;
    ``"grad"`` uses repeated ``torch.autograd.grad`` calls; and
    ``"gradient_difference"`` differences the selected autograd or network
    gradient. Set
    ``uniform_grid_batch_size`` to batch all initial uniform-grid derivatives
    on the selected Torch device before native extraction. Returns the native
    ``SurfaceMesh``.
    """
    adapter = TorchFieldAdapter(
        model,
        context,
        device=device,
        dtype=dtype,
        derivative_mode=derivative_mode,
        gradient_mode=gradient_mode,
        hessian_backend=hessian_backend,
        finite_difference_step=finite_difference_step,
    )
    selected_options = options if options is not None else SurfaceOptions()
    if uniform_grid_batch_size > 0:
        adapter.precompute_uniform_grid(
            bounds, selected_options.nx, selected_options.ny, selected_options.nz,
            uniform_grid_batch_size,
        )
    return extract_height_ridges_from_derivatives(
        adapter.gradient,
        adapter.hessian,
        bounds,
        selected_options,
        adapter.value,
    )
