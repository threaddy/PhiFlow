import numbers
import warnings
from functools import wraps, partial
from typing import List, Callable, Tuple, Union
from packaging import version

import jax
import jax.numpy as jnp
import jax.scipy as scipy
import numpy as np
from jax import random
from jax.core import Tracer

if version.parse(jax.__version__) >= version.parse('0.2.20'):
    from jax.experimental.sparse import BCOO, COO, CSR, CSC

from phi.math import DType
from phi.math.backend import Backend, ComputeDevice
from phi.math.backend._backend import combined_dim, SolveResult, PHI_LOGGER, TensorType
from ..math.backend._dtype import to_numpy_dtype, from_numpy_dtype


from jax.config import config
config.update("jax_enable_x64", True)


class JaxBackend(Backend):

    def __init__(self):
        devices = []
        for device_type in ['cpu', 'gpu', 'tpu']:
            try:
                for jax_dev in jax.devices(device_type):
                    devices.append(ComputeDevice(self, device_type.upper(), jax_dev.platform.upper(), -1, -1, f"id={jax_dev.id}", jax_dev))
            except RuntimeError as err:
                pass  # this is just Jax not finding anything. jaxlib.xla_client._get_local_backends() could help but isn't currently available on GitHub actions
        Backend.__init__(self, "Jax", devices, devices[-1])
        try:
            self.rnd_key = jax.random.PRNGKey(seed=0)
        except RuntimeError as err:
            warnings.warn(f"{err}", RuntimeWarning)
            self.rnd_key = None

    def prefers_channels_last(self) -> bool:
        return True

    def requires_fixed_shapes_when_tracing(self) -> bool:
        return True

    def _check_float64(self):
        if self.precision == 64:
            if not jax.config.read('jax_enable_x64'):
                jax.config.update('jax_enable_x64', True)
            assert jax.config.read('jax_enable_x64'), "FP64 is disabled for Jax."

    def seed(self, seed: int):
        self.rnd_key = jax.random.PRNGKey(seed)

    def as_tensor(self, x, convert_external=True):
        self._check_float64()
        if self.is_tensor(x, only_native=convert_external):
            array = x
        else:
            array = jnp.array(x)
        # --- Enforce Precision ---
        if not isinstance(array, numbers.Number):
            if self.dtype(array).kind == float:
                array = self.to_float(array)
            elif self.dtype(array).kind == complex:
                array = self.to_complex(array)
        return array

    def is_module(self, obj):
        return False

    def is_tensor(self, x, only_native=False):
        if isinstance(x, jnp.ndarray) and not isinstance(x, np.ndarray):  # NumPy arrays inherit from Jax arrays
            return True
        if isinstance(x, jnp.bool_) and not isinstance(x, np.bool_):
            return True
        if self.is_sparse(x):
            return True
        # --- Above considered native ---
        if only_native:
            return False
        # --- Non-native types ---
        if isinstance(x, np.ndarray):
            return True
        if isinstance(x, np.bool_):
            return True
        if isinstance(x, (numbers.Number, bool)):
            return True
        if isinstance(x, (tuple, list)):
            return all([self.is_tensor(item, False) for item in x])
        return False

    def is_sparse(self, x) -> bool:
        return isinstance(x, (COO, BCOO, CSR, CSC))

    def is_available(self, tensor):
        return not isinstance(tensor, Tracer)

    def numpy(self, tensor):
        if isinstance(tensor, COO):
            raise NotImplementedError
        elif isinstance(tensor, BCOO):
            indices = np.array(tensor.indices)
            values = np.array(tensor.data)
            indices = indices[..., 0], indices[..., 1]
            assert values.ndim == 1, f"Cannot convert batched COO to NumPy"
            from scipy.sparse import coo_matrix
            return coo_matrix((values, indices), shape=self.staticshape(tensor))
        elif isinstance(tensor, CSR):
            raise NotImplementedError
        elif isinstance(tensor, CSC):
            raise NotImplementedError
        else:
            return np.array(tensor)

    def to_dlpack(self, tensor):
        from jax import dlpack
        return dlpack.to_dlpack(tensor)

    def from_dlpack(self, capsule):
        from jax import dlpack
        return dlpack.from_dlpack(capsule)

    def copy(self, tensor, only_mutable=False):
        return jnp.array(tensor, copy=True)

    def get_device(self, tensor: TensorType) -> ComputeDevice:
        return self.get_device_by_ref(tensor.device())

    def allocate_on_device(self, tensor: TensorType, device: ComputeDevice) -> TensorType:
        return jax.device_put(tensor, device.ref)

    sqrt = staticmethod(jnp.sqrt)
    exp = staticmethod(jnp.exp)
    softplus = staticmethod(jax.nn.softplus)
    sin = staticmethod(jnp.sin)
    arcsin = staticmethod(jnp.arcsin)
    cos = staticmethod(jnp.cos)
    arccos = staticmethod(jnp.arccos)
    tan = staticmethod(jnp.tan)
    arctan = staticmethod(np.arctan)
    arctan2 = staticmethod(np.arctan2)
    sinh = staticmethod(np.sinh)
    arcsinh = staticmethod(np.arcsinh)
    cosh = staticmethod(np.cosh)
    arccosh = staticmethod(np.arccosh)
    tanh = staticmethod(np.tanh)
    arctanh = staticmethod(np.arctanh)
    log = staticmethod(jnp.log)
    log2 = staticmethod(jnp.log2)
    log10 = staticmethod(jnp.log10)
    isfinite = staticmethod(jnp.isfinite)
    isnan = staticmethod(jnp.isnan)
    isinf = staticmethod(jnp.isinf)
    abs = staticmethod(jnp.abs)
    sign = staticmethod(jnp.sign)
    round = staticmethod(jnp.round)
    ceil = staticmethod(jnp.ceil)
    floor = staticmethod(jnp.floor)
    flip = staticmethod(jnp.flip)
    stop_gradient = staticmethod(jax.lax.stop_gradient)
    transpose = staticmethod(jnp.transpose)
    equal = staticmethod(jnp.equal)
    tile = staticmethod(jnp.tile)
    stack = staticmethod(jnp.stack)
    concat = staticmethod(jnp.concatenate)
    maximum = staticmethod(jnp.maximum)
    minimum = staticmethod(jnp.minimum)
    clip = staticmethod(jnp.clip)
    shape = staticmethod(jnp.shape)
    staticshape = staticmethod(jnp.shape)
    imag = staticmethod(jnp.imag)
    real = staticmethod(jnp.real)
    conj = staticmethod(jnp.conjugate)
    einsum = staticmethod(jnp.einsum)
    cumsum = staticmethod(jnp.cumsum)

    def nonzero(self, values):
        result = jnp.nonzero(values)
        return jnp.stack(result, -1)

    def vectorized_call(self, f, *args, output_dtypes=None, **aux_args):
        batch_size = self.determine_size(args, 0)
        args = [self.tile_to(t, 0, batch_size) for t in args]
        def f_positional(*args):
            return f(*args, **aux_args)
        vec_f = jax.vmap(f_positional, 0, 0)
        return vec_f(*args)

    def jit_compile(self, f: Callable) -> Callable:
        def run_jit_f(*args):
            # print(jax.make_jaxpr(f)(*args))
            PHI_LOGGER.debug(f"JaxBackend: running jit-compiled '{f.__name__}' with shapes {[self.shape(arg) for arg in args]} and dtypes {[self.dtype(arg) for arg in args]}")
            return self.as_registered.call(jit_f, *args, name=f"run jit-compiled '{f.__name__}'")

        run_jit_f.__name__ = f"Jax-Jit({f.__name__})"
        jit_f = jax.jit(f, device=self._default_device.ref)
        return run_jit_f

    def block_until_ready(self, values):
        if hasattr(values, 'block_until_ready'):
            values.block_until_ready()
        if isinstance(values, (tuple, list)):
            for v in values:
                self.block_until_ready(v)

    def jacobian(self, f, wrt: Union[tuple, list], get_output: bool, is_f_scalar: bool):
        if get_output:
            jax_grad_f = jax.value_and_grad(f, argnums=wrt, has_aux=True)
            @wraps(f)
            def unwrap_outputs(*args):
                args = [self.to_float(arg) if self.dtype(arg).kind in (bool, int) else arg for arg in args]
                (_, output_tuple), grads = jax_grad_f(*args)
                return (*output_tuple, *grads)
            return unwrap_outputs
        else:
            @wraps(f)
            def nonaux_f(*args):
                loss, output = f(*args)
                return loss
            return jax.grad(nonaux_f, argnums=wrt, has_aux=False)

    def custom_gradient(self, f: Callable, gradient: Callable, get_external_cache: Callable = None, on_call_skipped: Callable = None) -> Callable:
        jax_fun = jax.custom_vjp(f)  # custom vector-Jacobian product (reverse-mode differentiation)

        def forward(*x):
            y = f(*x)
            return y, (x, y)

        def backward(x_y, dy):
            x, y = x_y
            dx = gradient(x, y, dy)
            return tuple(dx)

        jax_fun.defvjp(forward, backward)
        return jax_fun

    def divide_no_nan(self, x, y):
        return jnp.where(y == 0, 0, x / y)
        # jnp.nan_to_num(x / y, copy=True, nan=0) covers up NaNs from before

    def random_uniform(self, shape, low, high, dtype: Union[DType, None]):
        self._check_float64()
        self.rnd_key, subkey = jax.random.split(self.rnd_key)

        dtype = dtype or self.float_type
        jdt = to_numpy_dtype(dtype)
        if dtype.kind == float:
            tensor = random.uniform(subkey, shape, minval=low, maxval=high, dtype=jdt)
        elif dtype.kind == complex:
            real = random.uniform(subkey, shape, minval=low.real, maxval=high.real, dtype=to_numpy_dtype(DType(float, dtype.precision)))
            imag = random.uniform(subkey, shape, minval=low.imag, maxval=high.imag, dtype=to_numpy_dtype(DType(float, dtype.precision)))
            return real + 1j * imag
        elif dtype.kind == int:
            tensor = random.randint(subkey, shape, low, high, dtype=jdt)
            if tensor.dtype != jdt:
                warnings.warn(f"Jax failed to sample random integers with dtype {dtype}, returned {tensor.dtype} instead.", RuntimeWarning)
        else:
            raise ValueError(dtype)
        return jax.device_put(tensor, self._default_device.ref)

    def random_normal(self, shape, dtype: DType):
        self._check_float64()
        self.rnd_key, subkey = jax.random.split(self.rnd_key)
        dtype = dtype or self.float_type
        return jax.device_put(random.normal(subkey, shape, dtype=to_numpy_dtype(dtype)), self._default_device.ref)

    def range(self, start, limit=None, delta=1, dtype: DType = DType(int, 32)):
        if limit is None:
            start, limit = 0, start
        return jnp.arange(start, limit, delta, to_numpy_dtype(dtype))

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        assert mode in ('constant', 'symmetric', 'periodic', 'reflect', 'boundary'), mode
        if mode == 'constant':
            constant_values = jnp.array(constant_values, dtype=value.dtype)
            return jnp.pad(value, pad_width, 'constant', constant_values=constant_values)
        else:
            if mode in ('periodic', 'boundary'):
                mode = {'periodic': 'wrap', 'boundary': 'edge'}[mode]
            return jnp.pad(value, pad_width, mode)

    def reshape(self, value, shape):
        return jnp.reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        if isinstance(value, (tuple, list)):
            assert axis == 0
            return sum(value[1:], value[0])
        return jnp.sum(value, axis=axis, keepdims=keepdims)

    def prod(self, value, axis=None):
        if not isinstance(value, jnp.ndarray):
            value = jnp.array(value)
        if value.dtype == bool:
            return jnp.all(value, axis=axis)
        return jnp.prod(value, axis=axis)

    def where(self, condition, x=None, y=None):
        if x is None or y is None:
            return jnp.argwhere(condition)
        return jnp.where(condition, x, y)

    def zeros(self, shape, dtype: DType = None):
        self._check_float64()
        return jax.device_put(jnp.zeros(shape, dtype=to_numpy_dtype(dtype or self.float_type)), self._default_device.ref)

    def zeros_like(self, tensor):
        return jax.device_put(jnp.zeros_like(tensor), self._default_device.ref)

    def ones(self, shape, dtype: DType = None):
        self._check_float64()
        return jax.device_put(jnp.ones(shape, dtype=to_numpy_dtype(dtype or self.float_type)), self._default_device.ref)

    def ones_like(self, tensor):
        return jax.device_put(jnp.ones_like(tensor), self._default_device.ref)

    def meshgrid(self, *coordinates):
        self._check_float64()
        coordinates = [self.as_tensor(c) for c in coordinates]
        return [jax.device_put(c, self._default_device.ref) for c in jnp.meshgrid(*coordinates, indexing='ij')]

    def linspace(self, start, stop, number):
        self._check_float64()
        return jax.device_put(jnp.linspace(start, stop, number, dtype=to_numpy_dtype(self.float_type)), self._default_device.ref)

    def linspace_without_last(self, start, stop, number):
        self._check_float64()
        return jax.device_put(jnp.linspace(start, stop, number, endpoint=False, dtype=to_numpy_dtype(self.float_type)), self._default_device.ref)

    def mean(self, value, axis=None, keepdims=False):
        return jnp.mean(value, axis, keepdims=keepdims)

    def log_gamma(self, x):
        return jax.lax.lgamma(self.to_float(x))

    def tensordot(self, a, a_axes: Union[tuple, list], b, b_axes: Union[tuple, list]):
        return jnp.tensordot(a, b, (a_axes, b_axes))

    def mul(self, a, b):
        # if scipy.sparse.issparse(a):  # TODO sparse?
        #     return a.multiply(b)
        # elif scipy.sparse.issparse(b):
        #     return b.multiply(a)
        # else:
            return Backend.mul(self, a, b)

    def mul_matrix_batched_vector(self, A, b):
        from jax.experimental.sparse import BCOO
        if isinstance(A, BCOO):
            return(A @ b.T).T
        return jnp.stack([A.dot(b[i]) for i in range(b.shape[0])])

    def get_diagonal(self, matrices, offset=0):
        result = jnp.diagonal(matrices, offset=offset, axis1=1, axis2=2)
        return jnp.transpose(result, [0, 2, 1])

    def while_loop(self, loop: Callable, values: tuple, max_iter: Union[int, Tuple[int, ...], List[int]]):
        if all(self.is_available(t) for t in values):
            return self.stop_gradient_tree(Backend.while_loop(self, loop, values, max_iter))
        if isinstance(max_iter, (tuple, list)):  # stack traced trajectory, unroll until max_iter
            values = self.stop_gradient_tree(values)
            trj = [values] if 0 in max_iter else []
            for i in range(1, max(max_iter) + 1):
                values = loop(*values)
                if i in max_iter:
                    trj.append(values)  # values are not mutable so no need to copy
            return self.stop_gradient_tree(self.stack_leaves(trj))
        else:
            if max_iter is None:
                cond = lambda vals: jnp.any(vals[0])
                body = lambda vals: loop(*vals)
                return jax.lax.while_loop(cond, body, values)
            else:
                cond = lambda vals: jnp.any(vals[1][0]) & (vals[0] < max_iter)
                body = lambda vals: (vals[0] + 1, loop(*vals[1]))
                return jax.lax.while_loop(cond, body, (self.as_tensor(0), values))[1]

    def max(self, x, axis=None, keepdims=False):
        return jnp.max(x, axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        return jnp.min(x, axis, keepdims=keepdims)

    def conv(self, value, kernel, zero_padding=True):
        assert kernel.shape[0] in (1, value.shape[0])
        assert value.shape[1] == kernel.shape[2], f"value has {value.shape[1]} channels but kernel has {kernel.shape[2]}"
        assert value.ndim + 1 == kernel.ndim
        # AutoDiff may require jax.lax.conv_general_dilated
        result = []
        for b in range(value.shape[0]):
            b_kernel = kernel[min(b, kernel.shape[0] - 1)]
            result_b = []
            for o in range(kernel.shape[1]):
                result_b.append(0)
                for i in range(value.shape[1]):
                    # result.at[b, o, ...].set(scipy.signal.correlate(value[b, i, ...], b_kernel[o, i, ...], mode='same' if zero_padding else 'valid'))
                    result_b[-1] += scipy.signal.correlate(value[b, i, ...], b_kernel[o, i, ...], mode='same' if zero_padding else 'valid')
            result.append(jnp.stack(result_b, 0))
        return jnp.stack(result, 0)

    def expand_dims(self, a, axis=0, number=1):
        for _i in range(number):
            a = jnp.expand_dims(a, axis)
        return a

    def cast(self, x, dtype: DType):
        if self.is_tensor(x, only_native=True) and from_numpy_dtype(x.dtype) == dtype:
            return x
        else:
            return jnp.array(x, to_numpy_dtype(dtype))

    def unravel_index(self, flat_index, shape):
        return jnp.stack(jnp.unravel_index(flat_index, shape), -1)

    def ravel_multi_index(self, multi_index, shape, mode: Union[str, int] = 'undefined'):
        if not self.is_available(shape):
            return Backend.ravel_multi_index(self, multi_index, shape, mode)
        mode = mode if isinstance(mode, int) else {'undefined': 'clip', 'periodic': 'wrap', 'clamp': 'clip'}[mode]
        idx_first = jnp.transpose(multi_index, (self.ndims(multi_index)-1,) + tuple(range(self.ndims(multi_index)-1)))
        result = jnp.ravel_multi_index(idx_first, shape, mode='wrap' if isinstance(mode, int) else mode)
        if isinstance(mode, int):
            outside = self.any((multi_index < 0) | (multi_index >= jnp.asarray(shape, dtype=multi_index.dtype)), -1)
            result = self.where(outside, mode, result)
        return result

    def gather(self, values, indices, axis: int):
        slices = [indices if i == axis else slice(None) for i in range(self.ndims(values))]
        return values[tuple(slices)]

    def batched_gather_nd(self, values, indices):
        values = self.as_tensor(values)
        indices = self.as_tensor(indices)
        assert indices.shape[-1] == self.ndims(values) - 2
        batch_size = combined_dim(values.shape[0], indices.shape[0])
        results = []
        for b in range(batch_size):
            b_values = values[min(b, values.shape[0] - 1)]
            b_indices = self.unstack(indices[min(b, indices.shape[0] - 1)], -1)
            results.append(b_values[b_indices])
        return jnp.stack(results)

    def repeat(self, x, repeats, axis: int, new_length=None):
        return jnp.repeat(x, self.as_tensor(repeats), axis, total_repeat_length=new_length)

    def std(self, x, axis=None, keepdims=False):
        return jnp.std(x, axis, keepdims=keepdims)

    def boolean_mask(self, x, mask, axis=0, new_length=None, fill_value=0):
        if new_length is None:
            slices = [mask if i == axis else slice(None) for i in range(len(x.shape))]
            return x[tuple(slices)]
        else:
            indices = jnp.argwhere(mask, size=new_length, fill_value=-1)[..., 0]
            valid = indices >= 0
            valid = valid[tuple([slice(None) if i == axis else None for i in range(len(x.shape))])]
            result = self.gather(x, jnp.maximum(0, indices), axis)
            return jnp.where(valid, result, fill_value)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        if isinstance(boolean_tensor, (tuple, list)):
            boolean_tensor = jnp.stack(boolean_tensor)
        return jnp.any(boolean_tensor, axis=axis, keepdims=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        if isinstance(boolean_tensor, (tuple, list)):
            boolean_tensor = jnp.stack(boolean_tensor)
        return jnp.all(boolean_tensor, axis=axis, keepdims=keepdims)

    def scatter(self, base_grid, indices, values, mode: str):
        base_grid, values = self.auto_cast(base_grid, values)
        batch_size = combined_dim(combined_dim(indices.shape[0], values.shape[0]), base_grid.shape[0])
        spatial_dims = tuple(range(base_grid.ndim - 2))
        dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(1,),  # channel dim of updates (batch dim removed)
                                                inserted_window_dims=spatial_dims,  # no idea what this does but spatial_dims seems to work
                                                scatter_dims_to_operand_dims=spatial_dims)  # spatial dims of base_grid (batch dim removed)
        scatter = jax.lax.scatter_add if mode == 'add' else jax.lax.scatter
        result = []
        for b in range(batch_size):
            b_grid = base_grid[b, ...]
            b_indices = indices[min(b, indices.shape[0] - 1), ...]
            b_values = values[min(b, values.shape[0] - 1), ...]
            result.append(scatter(b_grid, b_indices, b_values, dnums))
        return jnp.stack(result)

    def histogram1d(self, values, weights, bin_edges):
        def unbatched_hist(values, weights, bin_edges):
            hist, _ = jnp.histogram(values, bin_edges, weights=weights)
            return hist
        return jax.vmap(unbatched_hist)(values, weights, bin_edges)

    def bincount(self, x, weights, bins: int):
        return jnp.bincount(x, weights=weights, minlength=bins, length=bins)

    def quantile(self, x, quantiles):
        return jnp.quantile(x, quantiles, axis=-1)

    def argsort(self, x, axis=-1):
        return jnp.argsort(x, axis)

    def searchsorted(self, sorted_sequence, search_values, side: str, dtype=DType(int, 32)):
        if self.ndims(sorted_sequence) == 1:
            return jnp.searchsorted(sorted_sequence, search_values, side=side).astype(to_numpy_dtype(dtype))
        else:
            return jax.vmap(partial(self.searchsorted, side=side, dtype=dtype))(sorted_sequence, search_values)

    def fft(self, x, axes: Union[tuple, list]):
        x = self.to_complex(x)
        if not axes:
            return x
        if len(axes) == 1:
            return jnp.fft.fft(x, axis=axes[0]).astype(x.dtype)
        elif len(axes) == 2:
            return jnp.fft.fft2(x, axes=axes).astype(x.dtype)
        else:
            return jnp.fft.fftn(x, axes=axes).astype(x.dtype)

    def ifft(self, k, axes: Union[tuple, list]):
        if not axes:
            return k
        if len(axes) == 1:
            return jnp.fft.ifft(k, axis=axes[0]).astype(k.dtype)
        elif len(axes) == 2:
            return jnp.fft.ifft2(k, axes=axes).astype(k.dtype)
        else:
            return jnp.fft.ifftn(k, axes=axes).astype(k.dtype)

    def dtype(self, array) -> DType:
        if isinstance(array, int):
            return DType(int, 32)
        if isinstance(array, float):
            return DType(float, 64)
        if isinstance(array, complex):
            return DType(complex, 128)
        if not isinstance(array, jnp.ndarray):
            array = jnp.array(array)
        return from_numpy_dtype(array.dtype)

    def matrix_solve_least_squares(self, matrix: TensorType, rhs: TensorType) -> Tuple[TensorType, TensorType, TensorType, TensorType]:
        solution, residuals, rank, singular_values = lstsq_batched(matrix, rhs)
        return solution, residuals, rank, singular_values

    def solve_triangular_dense(self, matrix, rhs, lower: bool, unit_diagonal: bool):
        matrix, rhs = self.auto_cast(matrix, rhs, int_to_float=True, bool_to_int=True)
        x = jax.lax.linalg.triangular_solve(matrix, rhs, lower=lower, unit_diagonal=unit_diagonal, left_side=True)
        return x

    def sparse_coo_tensor(self, indices: Union[tuple, list], values, shape: tuple):
        return BCOO((values, indices), shape=shape)


lstsq_batched = jax.vmap(jnp.linalg.lstsq)  # map first dimension, required for JaxBackend.matrix_solve_least_squares()
