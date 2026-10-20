"""ctypes loader for the scan kernels; compiles them on first use."""
import ctypes
import os
import subprocess
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "scan.c")
_VARIANTS = {
    "simd": ["-O3", "-mcpu=native", "-DUSE_NEON"],
    "scalar": ["-O3", "-mcpu=native", "-fno-vectorize", "-fno-slp-vectorize"],
    "i8": ["-O3", "-mcpu=native", "-DUSE_I8"],
}
_libs = {}


def _lib(variant):
    if variant in _libs:
        return _libs[variant]
    so = os.path.join(_HERE, f"libscan_{variant}.so")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(_SRC):
        subprocess.check_call(["cc", "-shared", "-fPIC", *_VARIANTS[variant], _SRC, "-o", so])
    lib = ctypes.CDLL(so)
    lib.scan_topk.restype = ctypes.c_int
    lib.scan_topk.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64,
                              ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    _libs[variant] = lib
    return lib


def scan_topk(X, ids, q, k, variant="simd"):
    """Top-k by inner product over rows `ids` of X (row-major, contiguous)."""
    lib = _lib(variant)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    out_i = np.empty(k, np.int64)
    out_s = np.empty(k, np.float32)
    n = lib.scan_topk(X.ctypes.data, X.shape[1], ids.ctypes.data, len(ids),
                      np.ascontiguousarray(q).ctypes.data, k, out_i.ctypes.data, out_s.ctypes.data)
    return out_i[:n], out_s[:n]


def quantize_i8(X):
    """Symmetric per-dataset int8 quantisation for unit-norm vectors."""
    s = 127.0 / max(1e-6, float(np.abs(X).max()))
    return np.clip(np.round(X * s), -127, 127).astype(np.int8), s
