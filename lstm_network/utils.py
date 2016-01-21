import numpy as np
from pycuda import gpuarray
import pycuda.elementwise as elementwise
from skcuda import install_headers


DEBUG = False

def decode(output_data, int_to_data):
    return int_to_data[np.argmax(output_data)]


def encode(x, data_to_int, noCh):
    input_int = data_to_int[x]
    encoded = np.zeros((noCh, 1))
    encoded[input_int] = 1
    return encoded

def debug(s):
    if DEBUG:
        print(s)

def decode_sequence(l, int_to_data):
    s = ""
    for c in l:
        s += decode(c, int_to_data)
    return s


def tanh(z_gpu):
    """
    Exponential integral with `n = 1` of complex arguments.

    Parameters
    ----------
    z_gpu : GPUArray
        Input matrix of shape `(m, n)`.

    Returns
    -------
    e_gpu : GPUArray
        GPUarrays containing the exponential integrals of
        the entries of `z_gpu`.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import scipy.special
    >>> import special
    >>> z = np.asarray(np.random.rand(4, 4)+1j*np.random.rand(4, 4), np.complex64)
    >>> z_gpu = gpuarray.to_gpu(z)
    >>> e_gpu = exp1(z_gpu)
    >>> e_sp = scipy.special.exp1(z)
    >>> np.allclose(e_sp, e_gpu.get())
    True
    """

    if z_gpu.dtype == np.complex64:
        args = 'pycuda::complex<float> *z, pycuda::complex<float> *e'
    elif z_gpu.dtype == np.complex128:
        args = 'pycuda::complex<double> *z, pycuda::complex<double> *e'
    else:
        raise ValueError('unsupported type')
    op = 'e[i] = tanh(z[i])'

    func = elementwise.ElementwiseKernel(args, op,
                                         options=["-I", install_headers],
                                         preamble='#include "cuSpecialFuncs.h"')

    e_gpu = gpuarray.empty_like(z_gpu)
    func(z_gpu, e_gpu)

    return e_gpu
