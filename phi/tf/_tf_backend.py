import numbers
from contextlib import contextmanager
from functools import wraps, partial
from typing import List, Callable, Tuple, Union

import keras
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.framework.errors_impl import NotFoundError

from ..math.backend._backend import combined_dim, TensorType
from ..math.backend._dtype import DType, to_numpy_dtype, from_numpy_dtype
from phi.math.backend import Backend, ComputeDevice, NUMPY
from ._tf_cuda_resample import resample_cuda, use_cuda


class TFBackend(Backend):

    def __init__(self):
        devices = [ComputeDevice(self, device.name, simple_device_type(device.device_type), device.memory_limit, -1, str(device), device.name) for device in device_lib.list_local_devices()]
        # Example refs: '/device:CPU:0'
        default_device_ref = '/' + os.path.basename(tf.zeros(()).device)
        default_device = None
        for device in devices:
            if device.ref == default_device_ref:
                default_device = device
        assert default_device is not None
        Backend.__init__(self, "TensorFlow", devices, default_device)

    def prefers_channels_last(self) -> bool:
        return True

    def _device_for(self, *values):
        devices = set(v.device for v in values if hasattr(v, 'device'))
        if len(devices) == 0:
            return tf.device(self._default_device.ref)
        elif len(devices) == 1:
            return tf.device(next(iter(devices)))
        else:
            return tf.device(self._default_device.ref)

    def seed(self, seed: int):
        tf.random.set_seed(seed)

    def is_module(self, obj):
        return isinstance(obj, keras.Model)

    def is_tensor(self, x, only_native=False):
        is_tf_tensor = tf.is_tensor(x) is True  # tf.is_tensor() can return non-bool values which indicates not a Tensor
        if only_native:
            return is_tf_tensor
        else:
            return is_tf_tensor or NUMPY.is_tensor(x, only_native=False)

    def is_sparse(self, x) -> bool:
        return isinstance(x, tf.SparseTensor)

    def as_tensor(self, x, convert_external=True):
        with tf.device(self._default_device.ref):
            if self.is_tensor(x, only_native=convert_external):
                return tf.identity(x)
            tensor = tf.convert_to_tensor(x)
            # --- Enforce Precision ---
            if not isinstance(tensor, numbers.Number):
                if isinstance(tensor, np.ndarray):
                    tensor = NUMPY.as_tensor(tensor)
                elif tensor.dtype.is_floating:
                    tensor = self.to_float(tensor)
            return tensor

    def is_available(self, tensor) -> bool:
        if self.is_tensor(tensor, only_native=True):
            return tf.executing_eagerly()
        else:
            return True

    def numpy(self, tensor):
        if self.is_sparse(tensor):
            indices = np.array(tensor.indices)
            values = np.array(tensor.values)
            indices = indices[..., 0], indices[..., 1]
            from scipy.sparse import coo_matrix
            return coo_matrix((values, indices), shape=self.staticshape(tensor))
        if tf.is_tensor(tensor):
            return tensor.numpy()
        return NUMPY.numpy(tensor)

    def to_dlpack(self, tensor):
        from tensorflow import experimental
        return experimental.dlpack.to_dlpack(tensor)

    def from_dlpack(self, capsule):
        from tensorflow import experimental
        with tf.device(self._default_device.ref):
            return experimental.dlpack.from_dlpack(capsule)

    def copy(self, tensor, only_mutable=False):
        if not only_mutable or tf.executing_eagerly():
            with tf.device(tensor.device):
                return tf.identity(tensor)
        else:
            return tensor

    def get_device(self, tensor: TensorType) -> ComputeDevice:
        device_name = '/' + os.path.basename(tensor.device)
        return self.get_device_by_ref(device_name)

    def allocate_on_device(self, tensor: TensorType, device: ComputeDevice) -> TensorType:
        with tf.device(device.ref):
            result = tf.identity(tensor)
            assert self.get_device(result) == device
            return result

    def vectorized_call(self, f, *args, output_dtypes=None, **aux_args):
        batch_size = self.determine_size(args, 0)
        args = [self.tile_to(t, 0, batch_size) for t in args]
        if output_dtypes is None:
            output0 = f(*[t[0] for t in args], **aux_args)  # Call f to determine its output signature.
            output_dtypes = tf.nest.map_structure(lambda x: x.dtype, output0)
        else:
            output_dtypes = tf.nest.map_structure(lambda dtype: to_numpy_dtype(dtype), output_dtypes)
        return tf.map_fn(lambda vals: f(*vals, **aux_args), tuple(args), fn_output_signature=output_dtypes)

    def jit_compile(self, f: Callable) -> Callable:
        compiled = tf.function(f)
        return lambda *args: self.as_registered.call(compiled, *args, name=f"run jit-compiled '{f.__name__}'")

    def custom_gradient(self, f: Callable, gradient: Callable, get_external_cache: Callable = None, on_call_skipped: Callable = None) -> Callable:
        @tf.custom_gradient
        def tf_function(*args, **kwargs):
            def grad(*grad_args):
                return gradient(args, y, grad_args)
            y = f(*args, **kwargs)
            return y, grad
        return tf_function

    def transpose(self, tensor, axes):
        with tf.device(tensor.device):
            return tf.transpose(tensor, perm=axes)

    def equal(self, x, y):
        with self._device_for(x, y):
            x, y = self.auto_cast(x, y)
            return tf.equal(x, y)

    def divide_no_nan(self, x, y):
        with self._device_for(x, y):
            x, y = self.auto_cast(x, y)
            return tf.math.divide_no_nan(x, y)

    def random_uniform(self, shape, low, high, dtype: Union[DType, None]):
        dtype = dtype or self.float_type
        tdt = to_numpy_dtype(dtype)
        with tf.device(self._default_device.ref):
            if dtype.kind != complex:
                return tf.random.uniform(shape, low, high, dtype=tdt)
            else:
                real = tf.cast(tf.random.uniform(shape, low.real, high.real, dtype=to_numpy_dtype(DType(float, dtype.precision))), tdt)
                imag = tf.cast(tf.random.uniform(shape, low.imag, high.imag, dtype=to_numpy_dtype(DType(float, dtype.precision))), tdt)
                return real + 1j * imag

    def random_normal(self, shape, dtype: DType):
        with tf.device(self._default_device.ref):
            return tf.random.normal(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def range(self, start, limit=None, delta=1, dtype: DType = DType(int, 32)):
        with tf.device(self._default_device.ref):
            return tf.range(start, limit, delta, to_numpy_dtype(dtype))

    def tile(self, value, multiples):
        with tf.device(value.device):
            if isinstance(multiples, (tuple, list)) and self.ndims(value) < len(multiples):
                value = self.expand_dims(value, axis=0, number=len(multiples) - self.ndims(value))
            return tf.tile(value, multiples)

    def repeat(self, x, repeats, axis: int, new_length=None):
        x = self.as_tensor(x)
        with tf.device(x.device):
            return tf.repeat(x, repeats, axis)

    def stack(self, values, axis=0):
        with self._device_for(*values):
            return tf.stack(values, axis=axis)

    def concat(self, values, axis):
        with self._device_for(*values):
            values = self.auto_cast(*values)
            return tf.concat(values, axis)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        if mode == 'boundary' and np.all(np.array(pad_width) <= 1):
            mode = 'symmetric'
        if mode in ('constant', 'symmetric', 'reflect'):
            with tf.device(value.device):
                constant_values = tf.cast(constant_values, value.dtype)
                return tf.pad(value, pad_width, mode.upper(), constant_values=constant_values)
        else:
            return NotImplemented

    def reshape(self, value, shape):
        with tf.device(value.device):
            return tf.reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        with tf.device(value.device):
            if self.dtype(value).kind == bool:
                value = self.to_int32(value)
            if axis is not None:
                if not isinstance(axis, int):
                    axis = list(axis)
            if isinstance(value, tf.SparseTensor):
                return tf.sparse.reduce_sum(value, axis=axis, keepdims=keepdims, output_is_sparse=False)
            if isinstance(value, (tuple, list)) and any([isinstance(x, tf.SparseTensor) for x in value]):
                result = value[0]
                for v in value[1:]:
                    result = tf.sparse.add(result, v, threshold=0)
                return result
            return tf.reduce_sum(value, axis=axis, keepdims=keepdims)

    def prod(self, value, axis=None):
        with tf.device(value.device):
            if axis is not None:
                if not isinstance(axis, int):
                    axis = list(axis)
            if value.dtype == bool:
                return tf.reduce_all(value, axis=axis)
            return tf.reduce_prod(value, axis=axis)

    def where(self, condition, x=None, y=None):
        with self._device_for(condition, x, y):
            x, y = self.auto_cast(x, y)
            condition = tf.cast(condition, tf.bool)
            return tf.where(condition, x, y)

    def nonzero(self, values):
        with tf.device(values.device):
            return tf.where(tf.not_equal(values, 0))

    def mean(self, value, axis=None, keepdims=False):
        with tf.device(value.device):
            if self.dtype(value).kind not in (float, complex):
                value = self.to_float(value)
            if axis is not None:
                if not isinstance(axis, int):
                    axis = list(axis)
            return tf.reduce_mean(value, axis, keepdims=keepdims)

    def grid_sample(self, grid, coordinates, extrapolation: str):
        assert extrapolation in ('undefined', 'zeros', 'boundary', 'periodic', 'symmetric', 'reflect'), extrapolation
        if use_cuda(grid):
            if self.staticshape(grid)[0] > self.staticshape(coordinates)[0]:
                assert self.staticshape(coordinates)[0] == 1
                coordinates = self.tile(coordinates, [self.staticshape(grid)[0], *[1] * (self.ndims(coordinates) - 1)])
            return resample_cuda(grid, coordinates, extrapolation)
        else:
            return NotImplemented

    def zeros(self, shape, dtype: DType = None):
        with tf.device(self._default_device.ref):
            return tf.zeros(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def zeros_like(self, tensor):
        with tf.device(self._default_device.ref):
            return tf.zeros_like(tensor)

    def ones(self, shape, dtype: DType = None):
        with tf.device(self._default_device.ref):
            return tf.ones(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def ones_like(self, tensor):
        with tf.device(self._default_device.ref):
            return tf.ones_like(tensor)

    def meshgrid(self, *coordinates):
        with tf.device(self._default_device.ref):
            result = tf.meshgrid(*coordinates, indexing='ij')
            return result

    def linspace(self, start, stop, number):
        with tf.device(self._default_device.ref):
            return self.to_float(tf.linspace(start, stop, number))

    def tensordot(self, a, a_axes: Union[tuple, list], b, b_axes: Union[tuple, list]):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b, bool_to_int=True)
            return tf.tensordot(a, b, (a_axes, b_axes))

    def mul_matrix_batched_vector(self, A, b):
        with self._device_for(A, b):
            if isinstance(A, tf.SparseTensor):
                result_T = tf.sparse.sparse_dense_matmul(A, tf.transpose(b))  # result shape contains unknown size
                result = tf.transpose(result_T)
                result.set_shape(tf.TensorShape([b.shape[0], A.shape[0]]))
                return result
            else:
                return tf.transpose(tf.matmul(A, b, transpose_b=True))

    def einsum(self, equation, *tensors):
        with self._device_for(*tensors):
            return tf.einsum(equation, *tensors)

    def cumsum(self, x, axis: int):
        with tf.device(x.device):
            return tf.cumsum(x, axis=axis, exclusive=False)

    def while_loop(self, loop: Callable, values: tuple, max_iter: Union[int, Tuple[int, ...], List[int]]):
        with self._device_for(*values):
            if isinstance(max_iter, (tuple, list)):  # stack traced trajectory, unroll until max_iter
                values = self.stop_gradient_tree(values)
                trj = [values] if 0 in max_iter else []
                for i in range(1, max(max_iter) + 1):
                    values = loop(*values)
                    if i in max_iter:
                        trj.append(values)  # values are not mutable so no need to copy
                    condition = values[0]
                    if self.is_available(condition) and not self.any(values[0]):
                        break
                trj.extend([trj[-1]] * (len(max_iter) - len(trj)))  # fill trj with final values
                return self.stop_gradient_tree(self.stack_leaves(trj))
            else:
                cond = lambda c, *vals: tf.reduce_any(tf.cast(c, tf.bool))
                return self.stop_gradient_tree(tf.while_loop(cond, loop, values, maximum_iterations=max_iter))

    def stop_gradient_tree(self, tree):
        return tf.nest.map_structure(tf.stop_gradient, tree)

    def abs(self, x):
        with tf.device(x.device):
            return tf.abs(x)

    def sign(self, x):
        with tf.device(x.device):
            return tf.sign(x)

    def round(self, x):
        with tf.device(x.device):
            return tf.round(x)

    def ceil(self, x):
        with tf.device(x.device):
            return tf.math.ceil(x)

    def floor(self, x):
        with tf.device(x.device):
            return tf.floor(x)

    def max(self, x, axis=None, keepdims=False):
        with tf.device(x.device):
            if isinstance(x, (tuple, list)):
                x = tf.stack(x)
            if x.dtype == tf.bool:
                return tf.cast(tf.reduce_max(tf.cast(x, tf.uint8), axis=axis, keepdims=keepdims), tf.bool)  # reduce_max allows no bool
            return tf.reduce_max(x, axis=axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        with tf.device(x.device):
            if isinstance(x, (tuple, list)):
                x = tf.stack(x)
            if x.dtype == tf.bool:
                return tf.cast(tf.reduce_min(tf.cast(x, tf.uint8), axis=axis, keepdims=keepdims), tf.bool)  # reduce_min allows no bool
            return tf.reduce_min(x, axis=axis, keepdims=keepdims)

    def maximum(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return tf.maximum(a, b)

    def minimum(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return tf.minimum(a, b)

    def clip(self, x, minimum, maximum):
        with self._device_for(x, minimum, maximum):
            x, minimum, maximum = self.auto_cast(x, minimum, maximum)
            return tf.clip_by_value(x, minimum, maximum)

    def sqrt(self, x):
        with tf.device(x.device):
            return tf.sqrt(x)

    def exp(self, x):
        with tf.device(x.device):
            return tf.exp(x)

    def softplus(self, x):
        with tf.device(x.device):
            return tf.math.softplus(x)

    def log_gamma(self, x):
        with tf.device(x.device):
            return tf.math.lgamma(self.to_float(x))

    def conv(self, value, kernel, zero_padding=True):
        with self._device_for(value, kernel):
            value = self.to_float(value)
            kernel = self.to_float(kernel)  # should use auto_cast but TensorFlow only supports DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32
            if zero_padding:
                value_padding = [[0, 0]] * 2 + [[s // 2, (s - 1) // 2] for s in kernel.shape[3:]]
                value = tf.pad(value, value_padding)
            convf = {3: partial(tf.nn.conv1d, stride=1),
                     4: partial(tf.nn.conv2d, strides=[1, 1, 1, 1]),
                     5: partial(tf.nn.conv3d, strides=[1, 1, 1, 1, 1])}[len(value.shape)]
            value = tf.transpose(value, [0, *range(2, self.ndims(value)), 1])  # could use data_format='NC...' but it's supported neither on CPU and for int tensors
            kernel = tf.transpose(kernel, [0, *range(3, self.ndims(kernel)), 2, 1])
            if kernel.shape[0] == 1:
                result = convf(value, kernel[0, ...], padding='VALID')
            else:
                result = []
                for b in range(kernel.shape[0]):
                    result.append(convf(value[b:b+1, ...], kernel[b], padding='VALID'))
                result = tf.concat(result, 0)
            result = tf.transpose(result, [0, self.ndims(result) - 1, *range(1, self.ndims(result) - 1)])
            return result

    def expand_dims(self, a, axis=0, number=1):
        a = self.as_tensor(a)
        with tf.device(a.device):
            if number == 0:
                return a
            for _i in range(number):
                a = tf.expand_dims(a, axis)
            return a

    def shape(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor.shape
        else:
            with tf.device(tensor.device):
                return tf.shape(tensor)

    def staticshape(self, tensor):
        if self.is_tensor(tensor, only_native=True):
            return tuple(tensor.shape.as_list())
        else:
            return np.shape(tensor)

    def gather(self, values, indices, axis: int):
        with self._device_for(values, indices):
            indices = indices % self.cast(self.shape(values)[axis], self.dtype(indices))
            return tf.gather(values, indices, axis=axis)

    def gather_by_component_indices(self, values, *component_indices):
        indices = self.stack(component_indices, -1)
        return tf.gather_nd(values, indices)

    def batched_gather_nd(self, values, indices):
        with self._device_for(values, indices):
            values_shape = self.staticshape(values)
            if values_shape[0] == 1 and self.staticshape(indices)[0] > 1:
                result = tf.gather_nd(values[0, ...], indices, batch_dims=0)
                return result
            if values_shape[0] > 1 and self.staticshape(indices)[0] == 1:
                indices = tf.tile(indices, [values_shape[0]] + [1] * (len(values_shape) - 1))
            return tf.gather_nd(values, indices, batch_dims=1)

    def unstack(self, tensor, axis=0, keepdims=False):
        with tf.device(tensor.device):
            unstacked = tf.unstack(tensor, axis=axis)
            if keepdims:
                unstacked = [self.expand_dims(c, axis=axis) for c in unstacked]
            return unstacked

    def std(self, x, axis=None, keepdims=False):
        with tf.device(x.device):
            if self.dtype(x).kind not in (float, complex):
                x = self.to_float(x)
            _mean, var = tf.nn.moments(x, axis, keepdims=keepdims)
            return tf.sqrt(var)

    def boolean_mask(self, x, mask, axis=0, new_length=None, fill_value=0):
        with self._device_for(x, mask):
            return tf.boolean_mask(x, mask, axis=axis)

    def isfinite(self, x):
        if self.dtype(x).kind in (bool, int):
            return self.ones(self.shape(x), dtype=DType(bool))
        with tf.device(x.device):
            return tf.math.is_finite(x)

    def isnan(self, x):
        if self.dtype(x).kind in (bool, int):
            return self.zeros(self.shape(x), dtype=DType(bool))
        with tf.device(x.device):
            return tf.math.is_nan(x)

    def isinf(self, x):
        if self.dtype(x).kind in (bool, int):
            return self.zeros(self.shape(x), dtype=DType(bool))
        with tf.device(x.device):
            return tf.math.is_inf(x)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        with tf.device(boolean_tensor.device):
            if self.dtype(boolean_tensor).kind != bool:
                boolean_tensor = tf.not_equal(boolean_tensor, 0)
            return tf.reduce_any(boolean_tensor, axis=axis, keepdims=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        with tf.device(boolean_tensor.device):
            if self.dtype(boolean_tensor).kind != bool:
                boolean_tensor = tf.not_equal(boolean_tensor, 0)
            return tf.reduce_all(boolean_tensor, axis=axis, keepdims=keepdims)

    def quantile(self, x, quantiles):
        import tensorflow_probability as tfp
        with tf.device(x.device):
            x = self.to_float(x)
            result = tfp.stats.percentile(x, quantiles * 100, axis=-1, interpolation='linear')
            return result

    def argsort(self, x, axis=-1):
        return tf.argsort(x, axis)

    def searchsorted(self, sorted_sequence, search_values, side: str, dtype=DType(int, 32)):
        return tf.searchsorted(sorted_sequence, search_values, side=side, out_type=to_numpy_dtype(dtype))

    def scatter(self, base_grid, indices, values, mode: str):
        with self._device_for(base_grid, indices, values):
            base_grid, values = self.auto_cast(base_grid, values)
            indices = self.as_tensor(indices)
            batch_size = combined_dim(combined_dim(indices.shape[0], values.shape[0]), base_grid.shape[0])
            scatter = tf.tensor_scatter_nd_add if mode == 'add' else tf.tensor_scatter_nd_update
            result = []
            for b in range(batch_size):
                b_grid = base_grid[b, ...]
                b_indices = indices[min(b, indices.shape[0] - 1), ...]
                b_values = values[min(b, values.shape[0] - 1), ...]
                result.append(scatter(b_grid, b_indices, b_values))
            return self.stack(result, axis=0)

    def histogram1d(self, values, weights, bin_edges):
        with self._device_for(values, weights, bin_edges):
            bin_count = self.staticshape(bin_edges)[-1] - 1
            bin_indices = tf.minimum(tf.searchsorted(bin_edges, values, side='right') - 1, bin_count - 1)  # ToDo this includes values outside
            hist = tf.math.bincount(bin_indices, weights=weights, minlength=bin_count, maxlength=bin_count, axis=-1)
            return hist

    def bincount(self, x, weights, bins: int):
        return tf.math.bincount(x, weights=weights, minlength=bins, maxlength=bins)

    def fft(self, x, axes: Union[tuple, list]):
        if not axes:
            return x
        x = self.to_complex(x)
        perm = (*[i for i in range(self.ndims(x)) if i not in axes], *axes)
        iperm = np.argsort(perm)
        with tf.device(x.device):
            if len(axes) == 1:
                return tf.transpose(tf.signal.fft(tf.transpose(x, perm)), iperm)
            elif len(axes) == 2:
                return tf.transpose(tf.signal.fft2d(tf.transpose(x, perm)), iperm)
            elif len(axes) == 3:
                return tf.transpose(tf.signal.fft3d(tf.transpose(x, perm)), iperm)
            else:
                for axis in axes:
                    x = self.fft(x, [axis])
                return x

    def ifft(self, k, axes: Union[tuple, list]):
        if not axes:
            return k
        k = self.to_complex(k)
        perm = (*[i for i in range(self.ndims(k)) if i not in axes], *axes)
        iperm = np.argsort(perm)
        with tf.device(k.device):
            if len(axes) == 1:
                return tf.transpose(tf.signal.ifft(tf.transpose(k, perm)), iperm)
            elif len(axes) == 2:
                return tf.transpose(tf.signal.ifft2d(tf.transpose(k, perm)), iperm)
            elif len(axes) == 3:
                return tf.transpose(tf.signal.ifft3d(tf.transpose(k, perm)), iperm)
            else:
                for axis in axes:
                    k = self.ifft(k, [axis])
                return k

    def imag(self, x):
        with tf.device(x.device):
            return tf.math.imag(x)

    def real(self, x):
        with tf.device(x.device):
            return tf.math.real(x)

    def conj(self, x):
        with tf.device(x.device):
            return tf.math.conj(x)

    def cast(self, x, dtype: DType):
        if not self.is_tensor(x, only_native=True):
            x = self.as_tensor(x, convert_external=True)
        if self.dtype(x) == dtype:
            return x
        else:
            with tf.device(x.device):
                return tf.cast(x, to_numpy_dtype(dtype))

    def unravel_index(self, flat_index, shape):
        idx_first = tf.unravel_index(flat_index, shape)
        return tf.transpose(idx_first, perm=tuple(range(1, self.ndims(flat_index)+1)) + (0,))

    def sin(self, x):
        with tf.device(x.device):
            return tf.math.sin(x)

    def arcsin(self, x):
        with tf.device(x.device):
            return tf.math.asin(x)

    def cos(self, x):
        with tf.device(x.device):
            return tf.math.cos(x)

    def arccos(self, x):
        with tf.device(x.device):
            return tf.math.acos(x)

    def tan(self, x):
        with tf.device(x.device):
            return tf.math.tan(x)

    def arctan(self, x):
        with tf.device(x.device):
            return tf.math.atan(x)

    def arctan2(self, y, x):
        y, x = self.auto_cast(y, x)
        with tf.device(x.device):
            return tf.math.atan2(y, x)

    def sinh(self, x):
        with tf.device(x.device):
            return tf.math.sinh(x)

    def arcsinh(self, x):
        with tf.device(x.device):
            return tf.math.asinh(x)

    def cosh(self, x):
        with tf.device(x.device):
            return tf.math.cosh(x)

    def arccosh(self, x):
        with tf.device(x.device):
            return tf.math.acosh(x)

    def tanh(self, x):
        with tf.device(x.device):
            return tf.math.tanh(x)

    def arctanh(self, x):
        with tf.device(x.device):
            return tf.math.atanh(x)

    def log(self, x):
        with tf.device(x.device):
            return tf.math.log(x)

    def sigmoid(self, x):
        with tf.device(x.device):
            return tf.math.sigmoid(x)

    def log2(self, x):
        with tf.device(x.device):
            return tf.math.log(x) / 0.6931471805599453094  # log(x) / log(2)

    def log10(self, x):
        with tf.device(x.device):
            return tf.math.log(x) / 2.3025850929940456840  # log(x) / log(10)

    def dtype(self, array) -> DType:
        if tf.is_tensor(array):
            dt = array.dtype.as_numpy_dtype
            return from_numpy_dtype(dt)
        else:
            return NUMPY.dtype(array)

    def sparse_coo_tensor(self, indices, values, shape):
        with self._device_for(indices, values):
            return tf.SparseTensor(indices=self.to_int64(indices), values=values, dense_shape=shape)

    def mul_coo_dense(self, indices, values, shape, dense):
        values, dense = self.auto_cast(values, dense)
        batch_size, nnz, channel_count = self.staticshape(values)
        if batch_size > 1:
            return Backend.mul_coo_dense(self, indices, values, shape, dense)
        indices = tf.cast(indices, np.int64)
        result = []
        for b in range(batch_size):
            b_result = []
            for c in range(channel_count):
                matrix = tf.SparseTensor(indices=indices[b], values=values[b, :, c], dense_shape=shape)
                try:
                    b_result.append(tf.sparse.sparse_dense_matmul(matrix, dense[b, :, c, :]))
                except NotFoundError:  # These data types are probably not supported by TensorFlow
                    return Backend.mul_coo_dense(self, indices, values, shape, dense)
            result.append(tf.stack(b_result))
        return tf.stack(result)

    def not_equal(self, x, y):
        with self._device_for(x, y):
            return ~self.equal(x, y)

    def greater_than(self, x, y):
        with self._device_for(x, y):
            x, y = self.auto_cast(x, y)
            return x > y

    def greater_or_equal(self, x, y):
        with self._device_for(x, y):
            x, y = self.auto_cast(x, y)
            return x >= y

    def add(self, a, b):
        with self._device_for(a, b):
            if isinstance(a, tf.SparseTensor) or isinstance(b, tf.SparseTensor):
                return tf.sparse.add(a, b, threshold=1e-5)
            else:
                return Backend.add(self, a, b)

    def sub(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return a - b

    def mul(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return a * b

    def div(self, numerator, denominator):
        with self._device_for(numerator, denominator):
            numerator, denominator = self.auto_cast(numerator, denominator)
            return numerator / denominator

    def pow(self, base, exp):
        with self._device_for(base, exp):
            base, exp = self.auto_cast(base, exp)
            return base ** exp

    def mod(self, dividend, divisor):
        with self._device_for(divisor, dividend):
            dividend, divisor = self.auto_cast(dividend, divisor)
            return dividend % divisor

    def and_(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return a & b

    def or_(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return a | b

    def xor(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return a ^ b

    def floordiv(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return a // b

    def jacobian(self, f, wrt: Union[tuple, list], get_output: bool, is_f_scalar: bool):
        @wraps(f)
        def eval_grad(*args):
            args = [self.as_tensor(arg, True) if i in wrt else arg for i, arg in enumerate(args)]
            args = [self.to_float(arg) if self.dtype(arg).kind in (bool, int) else arg for arg in args]
            wrt_args = [arg for i, arg in enumerate(args) if i in wrt]
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                for arg in wrt_args:
                    assert arg.dtype in (tf.float16, tf.float32, tf.float64, tf.complex64, tf.complex128), f"Gradients can only be computed for float or complex tensors but got {arg.dtype} for argument with shape {arg.shape}"
                    tape.watch(arg)
                loss, output = f(*args)
            if self.prod(tf.shape(loss)) == 1:
                grads = list(self.as_registered.call(tape.gradient, loss, wrt_args, name=f"Backpropagation"))
            else:
                grads = list(self.as_registered.call(tape.jacobian, loss, wrt_args, name=f"Backpropagation"))
            assert None not in grads, f"Gradient could not be computed for wrt argument {grads.index(None)} (argument {wrt[grads.index(None)]}) with shape {wrt_args[grads.index(None)].shape}. TensorFlow returned gradient=None."
            return (*output, *grads) if get_output else grads
        return eval_grad

    def stop_gradient(self, value):
        with self._device_for(value):
            return tf.stop_gradient(value)

    def matrix_solve_least_squares(self, matrix: TensorType, rhs: TensorType) -> Tuple[TensorType, TensorType, TensorType, TensorType]:
        with self._device_for(matrix, rhs):
            solution = tf.linalg.lstsq(matrix, rhs)
        return solution, None, None, None

    def solve_triangular_dense(self, matrix, rhs, lower: bool, unit_diagonal: bool):
        matrix, rhs = self.auto_cast(matrix, rhs, int_to_float=True, bool_to_int=True)
        rhs = self.expand_dims(rhs, -1)
        if unit_diagonal:
            diag = np.diag(np.ones((self.staticshape(matrix)[-1],)))
            matrix = self.where(diag, diag, matrix)
        result = tf.linalg.triangular_solve(matrix, rhs, lower=lower)
        return result[..., 0]

    def get_diagonal(self, matrices, offset=0):
        with self._device_for(matrices):
            matrices = tf.transpose(matrices, [0, 3, 1, 2])
            result = tf.linalg.diag_part(matrices, k=offset)
            return tf.transpose(result, [0, 2, 1])


_TAPES = []


def simple_device_type(t: str):
    return t[len('XLA_'):] if t.startswith('XLA_') else t
