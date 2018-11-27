import numpy.ctypeslib as npct
import numpy as np
if __name__ == "__main__":
    im = np.zeros((10),dtype=np.int32)
    array_1d_int32 = npct.ndpointer(dtype=np.int32,ndim=1,flags=['CONTIGUOUS'])
    dll = npct.load_library("MIC.dll",".")
    mic = dll.mic
    mic.restype = None
    mic.argtypes = [array_1d_int32]

    if not im.flags['C_CONTIGUOUS']:
        im = np.ascontiguousarray(im,dtype=im.dtype)
    mic(im)
    print(im)
