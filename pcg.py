import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
def pcg(A: sp.csr_matrix, b: np.array, tol: float = 1e-5, maxiter: int = 100, M1: sp.csr_matrix=None, M2: sp.csr_matrix=None, x0: np.array = None) -> (np.array,int,int):
    """
    PCG   Preconditioned Conjugate Gradients Method.
       X = PCG(A,B) attempts to solve the system of linear equations A*X=B for
       X. The N-by-N coefficient matrix A must be symmetric and positive
       definite and the right hand side column vector B must have length N.

       X = PCG(AFUN,B) accepts a function handle AFUN instead of the matrix A.
       AFUN(X) accepts a vector input X and returns the matrix-vector product
       A*X. In all of the following syntaxes, you can replace A by AFUN.

       X = PCG(A,B,TOL) specifies the tolerance of the method. If TOL is []
       then PCG uses the default, 1e-6.

       X = PCG(A,B,TOL,MAXIT) specifies the maximum number of itations. If
       MAXIT is [] then PCG uses the default, min(N,20).

       X = PCG(A,B,TOL,MAXIT,M) and X = PCG(A,B,TOL,MAXIT,M1,M2) use symmetric
       positive definite preconditioner M or M=M1*M2 and effectively solve the
       system inv(M)*A*X = inv(M)*B for X. If M is [] then a preconditioner
       is not applied. M may be a function handle MFUN returning M\X.

       X = PCG(A,B,TOL,MAXIT,M1,M2,X0) specifies the initial guess. If X0 is
       [] then PCG uses the default, an all zero vector.

       [X,FLAG] = PCG(A,B,...) also returns a convergence FLAG:
        0 PCG converged to the desired tolerance TOL within MAXIT itations
        1 PCG itated MAXIT times but did not converge.
        2 preconditioner M was ill-conditioned.
        3 PCG stagnated (two consecutive itates were the same).
        4 one of the scalar quantities calculated during PCG became too
          small or too large to continue computing.

       [X,FLAG,RELRES] = PCG(A,B,...) also returns the relative residual
       NORM(B-A*X)/NORM(B). If FLAG is 0, then RELRES <= TOL.

       [X,FLAG,RELRES,ITER] = PCG(A,B,...) also returns the itation number
       at which X was computed: 0 <= ITER <= MAXIT.

       [X,FLAG,RELRES,ITER,RESVEC] = PCG(A,B,...) also returns a vector of the
       estimated residual norms at each itation including NORM(B-A*X0).

       Example:
          n1 = 21; A = gallery('moler',n1);  b1 = A*ones(n1,1);
          tol = 1e-6;  maxit = 15;  M = diag([10:-1:1 1 1:10]);
          [x1,flag1,rr1,it1,rv1] = pcg(A,b1,tol,maxit,M);
       Or use this parameterized matrix-vector product function:
          afun = @(x,n)gallery('moler',n)*x;
          n2 = 21; b2 = afun(ones(n2,1),n2);
          [x2,flag2,rr2,it2,rv2] = pcg(@(x)afun(x,n2),b2,tol,maxit,M);

       Class support for inputs A,B,M1,M2,X0 and the output of AFUN:
          float: double

       See also BICG, BICGSTAB, BICGSTABL, CGS, GMRES, LSQR, MINRES, QMR,
       SYMMLQ, TFQMR, ICHOL, FUNCTION_HANDLE.

       Copyright 1984-2013 The MathWorks, Inc.
       """
    n = A.shape[0]
    n2b = np.linalg.norm(b)
    if n2b == 0:
        return np.zeros((n)),0,0
    if x0 == None:
        x = np.zeros((n))
    else:
        x = x0
    flag = 1
    it = 0
    xmin = x                          # Iterate which has minimal residual so far
    imin = 0                          # Iteration at which xmin was computed
    tolb = tol * n2b                  # Relative tolerance
    r = b - A.dot(x.transpose())
    normr = np.linalg.norm(r)                  # Norm of residual
    normr_act = normr
    eps = 2.2204e-16

    if normr <= tolb:
        return x,0,0

    normrmin = normr
    rho = 1
    stag = 0
    moresteps = 0
    maxmsteps = min(n // 50, 5, n - maxiter)
    maxstagsteps = 3
    ii = 0
    while ii < maxiter:
        if M1 != None:
            y = spsolve(M1,r)
        else:
            y = r

        if M2 != None:
            z = spsolve(M2,y)
        else:
            z = y
        rho1 = rho
        rho = r.dot(z)
        if (rho == 0) or np.isinf(rho):
            flag = 4
            break
        if (ii == 0):
            p = z
        else:
            beta = rho / rho1
            if ((beta == 0) or np.isinf(beta)):
                flag = 4
                break
            p = z + beta * p
        q = A.dot(p)
        pq = p.dot(q)
        if (pq <= 0) or np.isinf(pq):
            flag = 4
            break
        else:
            alpha = rho / pq
        # Check for stagnation of the method
        if norm(p) * abs(alpha) < eps * norm(x):
            stag = stag + 1
        else:
            stag = 0
        
        x = x + alpha * p             # form new itate
        r = r - alpha * q
        normr = norm(r)
        normr_act = normr
        # check for convergence
        if normr <= tolb or stag >= maxstagsteps or moresteps:
            r = b - A.dot(x)
            normr_act = norm(r)
            if normr_act <= tolb:
                flag = 0
                it = ii
                break
            else:
                if stag >= maxstagsteps and moresteps == 0:
                    stag = 0
                moresteps = moresteps + 1
                if moresteps >= maxmsteps:
                    flag = 3
                    it = ii
                    break
        if normr_act < normrmin:      # update minimal norm quantities
            normrmin = normr_act
            xmin = x
            imin = ii
        if stag >= maxstagsteps:
            flag = 3
            break
        ii += 1
    # returned solution is first with minimal residual
    if flag:
        r_comp = b - A.dot(xmin)
        if norm(r_comp) <= normr_act:
            x = xmin
            it = imin
        else:
            it = ii
    return x,flag,it