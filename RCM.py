import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d
from scipy.sparse.linalg import cg,spsolve,inv
from scipy.sparse import spdiags,diags
import scipy.sparse as sp
import os
from scipy import io
import cv2
import time
import pcg_solver
from pcg import pcg
import ichol
import copy
from ctypes import *
import numpy.ctypeslib as npct
def solveLinearEquation( IN, wx, wy, lbda, need_pcg=True):
    r, c = IN.shape
    k = r * c
    tempx = np.roll(wx,1,1)
    tempy = np.roll(wy,1,0)
    dxa = tempx + wx
    dya = tempy + wy
    D = 1 + (lbda * (dxa + dya)).transpose().reshape(-1)

    tempx = np.zeros((r,c))
    tempy = np.zeros((r,c))
    tempx[:,0] = wx[:,-1]
    tempy[0,:] = wy[-1,:]
    dxd1 = -lbda *tempx.transpose().reshape(-1)
    dyd1 = -lbda *tempy.transpose().reshape(-1)
    wx[:, -1] = 0
    wy[-1,:] = 0
    dxd2 = -lbda * wx.transpose().reshape(-1)
    dyd2 = -lbda * wy.transpose().reshape(-1)
    # Ax = spdiags(dxd2, -r, k, k)
    # Ay = spdiags(dyd2, -1, k, k)
    Ax = spdiags(np.stack([dxd1, dxd2]), [-k + r, -r], k, k)
    Ay = spdiags(np.stack([dyd1, dyd2]), [-r + 1, -1], k, k)

    A = (Ax + Ay) + (Ax + Ay).transpose() + diags(D)
    b = IN.transpose().reshape(-1)
    if need_pcg:
        #L = ichol(A, struct('michol', 'on'));
        #L = cholesky(A,lower=True)

        # solver = pcg_solver.PreConditionedConjugateGradientSolver("jacobi", A,IN.reshape(-1), np.zeros((k)),max_iter = 150,tol = 0.1)
        # tout, flag,it = solver.solve()
        #L = diags(A.diagonal(0))
        # array_1d_int32 = npct.ndpointer(dtype=np.int32, ndim=1, flags=['CONTIGUOUS'])
        # array_1d_float32 = npct.ndpointer(dtype=np.float32, ndim=1, flags=['CONTIGUOUS'])
        # dll = npct.load_library("MIC.dll", ".")
        # mic = dll.mic
        # mic.restype = None
        # mic.argtypes = [array_1d_int32,array_1d_int32,array_1d_float32,c_int32]
        # L = A.astype(np.float32)
        # mic(L.indptr,L.indices,L.data,k)
        # L = sp.triu(L).transpose()
        #tout, flag, _ = pcg(A, b, tol=0.1,maxiter=50,M1=L)
        tout, flag = cg(A, b, tol=0.1,maxiter=100)
        #print(flag)
    else:
        tout = spsolve(A,b)
    OUT = np.reshape(tout, (c, r)).transpose()
    return OUT


def RCM(image_path,model=None,need_pcg=True,fixedH=400):
    I = cv2.imread(image_path)
    I = I / 255.0
    #I = I[::10,::10,:]
    #################### Param Settings #######################
    alpha = 1
    eps = 0.001
    win = 5
    ratioMax = 7

    ######################## Camear Model ###################
    # we use beta-gamma model as default
    if not model:
        model = 'beta-gamma'
    model = model.lower()
    if model == 'preferred':
        a, b, c = 4.3536,1.2854,0.1447
        g = lambda I, k: \
            I * np.power(k,a*c) / (np.power(I,1.0/c) * (np.power(k,a) - 1) + 1) ** c
    elif model == 'beta-gamma':
        a,b = -0.3293, 1.1258
        f = lambda x: np.exp((1 - x**a) * b)
        g = lambda I,k: (I ** (k**a)) * f(k)
    elif model == "affine":
        alpha, beta, gamma = -0.1797,1.2076,0.3988
        g = lambda I,k: I *(k ** gamma) + alpha * (1-k **gamma)
    elif model == 'beta':
        c = 0.4800
        g = lambda k:I * k ** c
    else:
        raise BaseException('unkown model: %s' %  model)

    ########## Exposure Ratio Map #####################

    # Initial Exposure Ratio Map
    t0 = np.max(I,-1)
    #t0 = io.loadmat('../t0.mat')['t0']
    M, N = t0.shape
    #print(M,N)
    # Exposure Ratio Map Refinement
    #t0 = t0[0:M:2,0:N:2]
    fx = fixedH / M
    t0 = cv2.resize(t0,dsize=None,fx=fx,fy=fx)
    #t0 = cv2.resize(t0, (N//2, M//2))
    # dt0_v = np.diff(t0, axis=0)
    # dt0_h = np.diff(t0,axis=1)
    dt0_v = np.diff(np.concatenate([t0,np.expand_dims(t0[0,:],0)],axis=0),axis=0)
    dt0_h = np.diff(np.concatenate([t0,np.expand_dims(t0[:,0],1)],axis=1),axis=1)
    #kernel = np.ones((win,),dtype=np.float32)
    gauker_h = convolve2d(dt0_h,np.ones((1,win)),mode='same')
    gauker_v = convolve2d(dt0_v,np.ones((win,1)),mode='same')
    #gauker_h = convolve1d(dt0_h,kernel,axis=1)
    #gauker_v = convolve1d(dt0_v, kernel, axis=0)

    W_h = 1. / (np.abs(gauker_h) * np.abs(dt0_h) + eps)
    W_v = 1. / (np.abs(gauker_v) * np.abs(dt0_v) + eps)

    T = solveLinearEquation(t0, W_h, W_v, alpha / 2,need_pcg)
    T = cv2.resize(T,(N,M),interpolation=cv2.INTER_CUBIC)
    kRatio = np.minimum(1 / T, ratioMax)
    kRatio  = np.stack([kRatio,kRatio,kRatio],axis=-1)
    # # Enhancement
    J = g(I, kRatio)
    J = np.maximum(0,np.minimum(1,J))
    return J


if __name__ == "__main__":
    single = True
    if single:
        start=time.time()
        image_path = '0.jpg'
        E = RCM(image_path,need_pcg=True)
        print("Time: %.2fs" % (time.time()-start,))
        E = (E*255).astype(np.uint8)
        cv2.imshow("origin",cv2.imread(image_path))
        cv2.imshow("enhance", E)
        cv2.imwrite("0_python.jpg",E)
        cv2.waitKey(0)
    else:
        test_dir = '../../Test_images_120/'
        save_dir = '../../Test_images_120_en/python/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        images_names = os.listdir(test_dir)
        for fixH in range(50,500,50):
            start = time.time()
            for image_name in images_names:
                image_path = os.path.join(test_dir,image_name)
                E = RCM(image_path, need_pcg=True,fixedH=fixH)
                E = (E * 255).astype(np.uint8)
                save_image_name = "%s_%d.jpg" %(image_name[:-4],fixH)
                cv2.imwrite(os.path.join(save_dir,save_image_name), E)
            t = time.time()-start
            print("Resolution: " + str(fixH) +', Total time consumed: ',str(t), 's, average time consumed: ', str(t/len(images_names)),'s')
