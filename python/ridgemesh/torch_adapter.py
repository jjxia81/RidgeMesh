"""PyTorch and GeoUDF adapters for the RidgeMesh native surfacer.

The GeoUDF model's learned gradient output is useful for its own reconstruction
pipeline, but RidgeMesh needs derivatives of the *scalar UDF itself*. This
adapter obtains both gradient and Hessian with ``torch.autograd``.
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
    first item is the UDF, or a mapping containing ``"udf"``. GeoUDF's native
    convention is a query tensor with shape ``(B, 3, M)``; this adapter uses
    ``(1, 3, 1)`` for each requested point.

    Set ``derivative_mode="finite_difference"`` to estimate both derivatives
    from central scalar-UDF differences. Derivatives are cached by point
    because the C++ surfacer asks for a Hessian and gradient at the same
    locations. For large grids, prefer analytic C++ callbacks or a future
    batched-sample API: both modes remain pointwise at the native callback
    boundary.
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
        if hessian_backend not in ("grad", "functional"):
            raise ValueError("hessian_backend must be 'grad' or 'functional'")
        if finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive")
        self.derivative_mode = derivative_mode
        self.hessian_backend = hessian_backend
        self.finite_difference_step = float(finite_difference_step)
        self._cache: Dict[
            Tuple[float, float, float],
            Tuple[Tuple[float, float, float], List[List[float]]],
        ] = {}

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

    def _evaluate_derivatives(self, x: float, y: float, z: float):
        key = (float(x), float(y), float(z))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self.derivative_mode == "finite_difference":
            result = self._evaluate_finite_difference_derivatives(key)
        else:
            result = self._evaluate_autograd_derivatives(key)
        self._cache[key] = result
        return result

    def _evaluate_autograd_derivatives(self, point_coordinates):
        import torch

        point = torch.tensor(point_coordinates, device=self.device, dtype=self.dtype, requires_grad=True)
        query = point.reshape(1, 3, 1)
        with torch.enable_grad():
            udf = self._udf_from_output(self.model(self.context, query))
            if udf.numel() != 1:
                raise ValueError("the model must return one scalar UDF value for a (1, 3, 1) query")
            scalar_udf = udf.reshape(())
            gradient = torch.autograd.grad(scalar_udf, point, create_graph=True)[0]
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
            else:
                hessian_rows = [
                    torch.autograd.grad(gradient[index], point, retain_graph=index < 2)[0]
                    for index in range(3)
                ]

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
        """Return the autograd gradient in the format expected by pybind11."""
        return self._evaluate_derivatives(x, y, z)[0]

    def hessian(self, x: float, y: float, z: float):
        """Return the autograd Hessian in the format expected by pybind11."""
        return self._evaluate_derivatives(x, y, z)[1]


def extract_torch_udf(
    model: Callable,
    context: Optional[Mapping[str, Any]],
    bounds: Bounds3D,
    options: Optional[SurfaceOptions] = None,
    *,
    device: Optional[str] = None,
    dtype=None,
    derivative_mode: str = "autograd",
    hessian_backend: str = "functional",
    finite_difference_step: float = 5e-3,
):
    """Extract a ridge/valley mesh from a PyTorch UDF such as GeoUDF.

    The model is called as ``model(context, query)`` with a GeoUDF-compatible
    query tensor of shape ``(1, 3, 1)``. Set ``derivative_mode`` to
    ``"finite_difference"`` to avoid second-order autograd. With autograd,
    ``hessian_backend="functional"`` uses ``torch.autograd.functional.hessian``;
    ``"grad"`` uses repeated ``torch.autograd.grad`` calls. Returns the native
    ``SurfaceMesh``.
    """
    adapter = TorchFieldAdapter(
        model,
        context,
        device=device,
        dtype=dtype,
        derivative_mode=derivative_mode,
        hessian_backend=hessian_backend,
        finite_difference_step=finite_difference_step,
    )
    return extract_height_ridges_from_derivatives(
        adapter.gradient,
        adapter.hessian,
        bounds,
        options if options is not None else SurfaceOptions(),
    )
