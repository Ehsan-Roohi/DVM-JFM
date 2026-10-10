import ctypes,json
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','avail','tp','ap','tv','av','ex')]
def available():
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise RuntimeError('Invalid native RAM sample')
    return m.avail/2**30
if __name__=='__main__':print(json.dumps({'native_available_GiB':available()}))
