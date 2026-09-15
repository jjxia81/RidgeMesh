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

    Derivatives are cached by point because the C++ surfacer asks for a Hessian
    and gradient at the same locations. For large grids, prefer analytic C++
    callbacks or a future batched-sample API: autograd calls here remain
    pointwise at the native callback boundary.
    """

    def __init__(
        self,
        model: Callable,
        context: Optional[Mapping[str, Any]] = None,
        *,
        device: Optional[str] = None,
        dtype=None,
        move_context: bool = True,
    ):
        import torch

        self.model = model
        self.device = torch.device(device) if device is not None else self._model_device(model)
        self.dtype = dtype if dtype is not None else torch.float32
        self.context = _move_tensors(context or {}, self.device) if move_context else (context or {})
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
        import torch

        key = (float(x), float(y), float(z))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        point = torch.tensor(key, device=self.device, dtype=self.dtype, requires_grad=True)
        query = point.reshape(1, 3, 1)
        with torch.enable_grad():
            udf = self._udf_from_output(self.model(self.context, query))
            if udf.numel() != 1:
                raise ValueError("the model must return one scalar UDF value for a (1, 3, 1) query")
            scalar_udf = udf.reshape(())
            gradient = torch.autograd.grad(scalar_udf, point, create_graph=True)[0]
            hessian_rows = [
                torch.autograd.grad(gradient[index], point, retain_graph=index < 2)[0]
                for index in range(3)
            ]

        gradient_value = tuple(float(value) for value in gradient.detach().cpu())
        hessian_value = [
            [float(value) for value in row.detach().cpu()]
            for row in hessian_rows
        ]
        result = (gradient_value, hessian_value)
        self._cache[key] = result
        return result

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
):
    """Extract a ridge/valley mesh from a PyTorch UDF such as GeoUDF.

    The model is called as ``model(context, query)`` with a GeoUDF-compatible
    query tensor of shape ``(1, 3, 1)``. Returns the native ``SurfaceMesh``.
    """
    adapter = TorchFieldAdapter(model, context, device=device, dtype=dtype)
    return extract_height_ridges_from_derivatives(
        adapter.gradient,
        adapter.hessian,
        bounds,
        options if options is not None else SurfaceOptions(),
    )
