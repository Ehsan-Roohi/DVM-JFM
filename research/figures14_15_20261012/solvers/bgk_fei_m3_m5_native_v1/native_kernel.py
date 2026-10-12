from pathlib import Path
import ctypes as C
import numpy as np
lib=C.CDLL(str(Path(__file__).with_name('kernel.so')))
ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
lib.step_bgk.argtypes=[C.c_int,C.c_int,ptr,ptr,ptr,ptr,ptr,C.c_double,C.c_double,ptr,C.POINTER(C.c_double)]
lib.step_bgk.restype=C.c_int
def advance(f,v,w,left,right,c,dt):
    if c['Pr']!=1 or c['mu1']!=1 or c['omega']!=1:raise ValueError('Native BGK physical contract mismatch')
    out=np.empty_like(f);defect=C.c_double()
    code=lib.step_bgk(c['nx'],c['nv'],f,v,w,np.ascontiguousarray(left),np.ascontiguousarray(right),dt,2*c['L']/c['nx'],out,C.byref(defect))
    if code:raise FloatingPointError(f'Native BGK guard {code}; collision defect {defect.value}')
    return out
