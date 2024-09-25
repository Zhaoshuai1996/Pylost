# coding=utf-8
# /*##########################################################################
#
# SWaRP: Speckle Wavefront Reconstruction Package 
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
#
# This file is part of the SWaRP Speckle Wavefront Reconstruction Package
# developed at the ESRF by the staff of BM05 as part of EUCALL WP7:PUUCA.
#
# This project has received funding from the European Union’s Horizon 2020 
# research and innovation programme under grant agreement No 654220.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/

import warnings

import numpy as np
import scipy.linalg as la
from numpy.linalg import matrix_rank
from scipy.linalg import solve_sylvester


def norm(x):
    return np.sqrt(np.sum(x ** 2))


def dop(m, n=None):
    """Generate a set of discrete orthonormal polynomials and their 
    derivatives.
    
    Parameters
    ----------
    m : array or int
        If an integer, the number of evenly spaced points in the 
        support. If a vector, the (arbitrarily spaced) points in the 
        support.
    n : int
        The number of basis functions (i.e. the order of the 
        polynomial). Default is the size of `m`.
    Returns
    -------
    P : m x n array
        The discrete polynomials
    dP : m x n array
        The derivatives
    rC : tuple
        (alpha,beta) coefficients for the three term recurrence 
        relationship

    Notes
    -----
    Directly transcribed from MATLAB code: DOPbox v1.0:
    Cite this as :

        @article{DBLP:journals/tim/OLearyH12,
         author    = {Paul O'Leary and
                      Matthew Harker},
         title     = {A Framework for the Evaluation of Inclinometer 
                      Data in the Measurement of Structures},
         journal   = {IEEE T. Instrumentation and Measurement},
         volume    = {61},
         number    = {5},
         year      = {2012},
         pages     = {1237-1251},
        }
        @inproceedings{olearyHarker2008B,
          Author = {O'Leary, Paul and Harker, Matthew},
          Title = {An Algebraic Framework for Discrete Basis Functions 
                   in Computer Vision},
          BookTitle = {IEEE Indian Conference on Computer Vision, 
                       Graphics and Image Processing},
          Address= {Bhubaneswar, Dec},
          Year = {2008} }
        Author : Matthew Harker
        Date : Nov. 29, 2011
        Version : 1.0
        ----------------------------------------------------------------
        (c) 2011, Harker, O'Leary, University of Leoben, Leoben, Austria
        email: automation@unileoben.ac.at
        url: automation.unileoben.ac.at
        ----------------------------------------------------------------
    """

    u = len(m)

    if u == 1:
        x = np.arange(-1, 2 / (m - 1))
    else:
        x = m
        m = len(x)

    if n is None:
        n = m

    # Generate the Basis

    # Generate the first two polynomials :
    p0 = np.ones(m) / np.sqrt(m)
    meanX = np.mean(x)
    p1 = x - meanX
    np1 = norm(p1)
    p1 /= np1

    # Compute the derivatives of the degree-1 polynomial :
    hm = np.sum(np.diff(x))
    h = np.sum(np.diff(p1))
    dp1 = (h / hm) * np.ones(m)

    # Initialize the basis function matrices :
    P = np.zeros((m, n))
    P[:, 0] = p0
    P[:, 1] = p1

    dP = np.zeros((m, n))
    dP[:, 1] = dp1

    # Setup storage for the coefficients of the three term relationship
    alphas = np.zeros(n)
    alphas[0] = 1 / np.sqrt(m)
    alphas[1] = 1 / np1

    betas = np.zeros(n)
    betas[1] = meanX

    for k in range(2, n):
        # Augment previous polynomial :
        pt = P[:, k - 1] * p1

        # 3-term recurrence
        beta0 = np.dot(P[:, k - 2], pt)
        pt -= P[:, k - 2] * beta0
        betas[2] = beta0

        # Complete reorthogonalization :
        beta = np.dot(P[:, :k].T, pt)
        pt -= np.dot(P[:, :k], beta)

        # Apply coefficients to recurrence formulas :
        alpha = 1 / np.sqrt(np.dot(pt, pt))
        alphas[k] = alpha
        P[:, k] = np.dot(alpha, pt)
        dP[:, k] = alpha * (dP[:, k - 1] * p1 + P[:, k - 1] * dp1
                            - dP[:, k - 2] * beta0 - np.dot(dP[:, :k], beta))

    recurrenceCoeffs = (alphas, betas)

    return P, dP, recurrenceCoeffs


def diff_local(x, ls, noBfs):
    """Generates a global matrix operator which implements the 
    computation of local differentials.
    
    Parameters
    ----------
       x : 1d-array
       A vector of co-ordinates at which to evaluate the differentials.
       Can be irregularly spaced.
       ls : int
       Support length. Should be an odd number. There is an exception up 
       to ls = 20 and ls = noPoints.
       In this case a full differentiating matrix is computed.
       noBfs : int
       Number of basis functions to use.
    Returns
    -------
       S : square array
       The local differential matrix, each dimension is len(x).
    Notes
    -----
    Local discrete orthogonal polynomials are used to generate the local 
    approximations for the dreivatives.
    Transcribed from MATLAB code: DOPbox: dopDiffLocal.m, v1.0:
    Author :  Matthew Harker and Paul O'Leary
    Date :    17. January 2012
    Version : 1.0
    (c) 2013 Matthew Harker and Paul O'Leary,
    Chair of Automation, University of Leoben, Leoben, Austria
    email: office@harkeroleary.org,
    url: www.harkeroleary.org
    """

    noPts = len(x)

    # Test the input paramaters

    # if option != "full":
    #     genSparse = True
    # else:
    #     genSparse = False

    # if mt > 1:
    #     raise ValueError('A column vector is expected for x')

    # Test the degree and support length for campatability
    if noBfs > ls:
        raise ValueError("The number of basis functions must be <= ls")

    if ls > 13:
        warnings.warn("With a support length greater than 13 there may be "
                      "problems with the Runge phenomena.")

    # Compute a full matrix
    if ls == noPts:
        Gt, dGt, _ = dop(x, noBfs)
        S = np.dot(dGt, Gt.T)
        rS = matrix_rank(S)
        if rS < noPts - 1:
            warnings.warn("The rank of S is " + str(rS) + " while x has n = "
                          + str(noPts) + " points.")
        return S

    # Test if the support length is compatible with the number of points 
    # requested.
    if noPts < ls:
        raise ValueError("The number of nodes n must be greater that the "
                         "support length ls")

    if ls % 2 == 0:
        raise ValueError("This function is only implemented for odd values of "
                         "ls.")

    # ------------------------------------------------------------------------
    vals = np.zeros((noPts, noPts))  # I think?

    # Determine the half length of ls this determine the upper ane lower 
    # positions of Si.
    ls2 = np.rint((ls + 1) / 2).astype(int)

    # Generate the top of Si
    Gt, dGt, _ = dop(x[np.arange(ls)], noBfs)
    Dt = np.dot(dGt, Gt.T)
    vals[:ls2, :ls] = Dt[:ls2, :]

    # Compute the strip diagonal entries
    noOnDiag = noPts - 2 * ls2
    for k in range(noOnDiag):
        Gt, dGt, _ = dop(x[range(k + 1, k + ls + 1)], noBfs)
        tdGt = dGt[ls2 - 1, :]
        dt = np.dot(tdGt, Gt.T)
        vals[k + ls2, k + 1:k + 1 + ls] = dt

    # Generate the bottom part of Si
    Gt, dGt, _ = dop(x[-ls:], noBfs)
    Dt = np.dot(dGt, Gt.T)
    vals[-ls2:, -ls:] = Dt[-ls2:, :]

    rS = matrix_rank(vals)

    if rS < noPts - 1:
        warnings.warn("The rank of S is " + str(rS) + " while x has n = "
                      + str(noPts) + " points.")

    return vals


def mrdivide(a, b):
    """Problem: C = A/B
       -> CB = A
    If B is square:
       -> C = A*inv(B)
    Otherwise:
       -> C*(B*B') = A*B'
       -> C = A*B'*inv(B*B')
    """

    A = np.asmatrix(a)
    B = np.asmatrix(b)
    dims = B.shape
    if dims[0] == dims[1]:
        return A * B.I
    else:
        return (A * B.T) * (B * B.T).I


def mldivide(a, b):
    dimensions = a.shape
    if dimensions[0] == dimensions[1]:
        return la.solve(a, b)
    else:
        return la.lstsq(a, b)[0]


def g2sSylvester(A, B, F, G, u, v):
    """Purpose : Solves the semi-definite Sylvester Equation of the form
      A'*A * Phi + Phi * B'*B - A'*F - G*B = 0,
      Where the null vectors of A and B are known to be
      A * u = 0
      B * v = 0

    Use (syntax):
      Phi = g2sSylvester( A, B, F, G, u, v )

    Input Parameters :
      A, B, F, G := Coefficient matrices of the Sylvester Equation
      u, v := Respective null vectors of A and B

    Return Parameters :
      Phi := The minimal norm solution to the Sylvester Equation

    Description and algorithms:
      The rank deficient Sylvester equation is solved by means of 
      Householder reflections and the Bartels-Stewart algorithm.  It 
      uses the MATLAB function "lyap", in reference to Lyapunov 
      Equations, a special case of the Sylvester Equation.
    """

    # Householder vectors
    m, n = len(u), len(v)

    u[0] += norm(u)
    u *= np.sqrt(2) / norm(u)

    v[0] += norm(v)
    v *= np.sqrt(2) / norm(v)

    # Apply householder updates
    A -= np.dot(np.dot(A, u), u.T)
    B -= np.dot(np.dot(B, v), v.T)
    F -= np.dot(np.dot(F, v), v.T)
    G -= np.dot(u, (np.dot(u.T, G)))

    # Solve the system of equations
    phi = np.zeros((m, n))
    phi[0, 1:] = mrdivide(G[0, :], B[:, 1:].T)
    phi[1:, 0] = mldivide(A[:, 1:], F[:, 0].T)
    phi[1:, 1:] = solve_sylvester(np.dot(A[:, 1:].T, A[:, 1:]),
                                  np.dot(B[:, 1:].T, B[:, 1:]),
                                  -np.dot(-A[:, 1:].T, F[:, 1:])
                                  + np.dot(G[1:, :], B[:, 1:]))

    # Invert the householder updates
    phi -= np.dot(u, (np.dot(u.T, phi)))
    phi -= np.dot(np.dot(phi, v), v.T)

    return phi


def g2s(x, y, Zx, Zy, N=3):
    """Purpose : Computes the Global Least Squares reconstruction of a surface
      from its gradient field.

    Use (syntax):
      Z = g2s( Zx, Zy, x, y )
      Z = g2s( Zx, Zy, x, y, N )

    Input Parameters :
      Zx, Zy := Components of the discrete gradient field
      x, y := support vectors of nodes of the domain of the gradient
      N := number of points for derivative formulas (default=3)

    Return Parameters :
      Z := The reconstructed surface

    Description and algorithms:
      The algorithm solves the normal equations of the Least Squares 
      cost function, formulated by matrix algebra:
      e(Z) = || D_y * Z - Zy ||_F^2 + || Z * Dx' - Zx ||_F^2
      The normal equations are a rank deficient Sylvester equation which 
      is solved by means of Householder reflections and the 
      Bartels-Stewart algorithm.
    """

    if Zx.shape != Zy.shape:
        raise ValueError("Gradient components must be the same size")

    if (np.asmatrix(Zx).shape[1] != len(x) or
            np.asmatrix(Zx).shape[0] != len(y)):
        raise ValueError("Support vectors must have the same size as the "
                         "gradient")

    m, n = Zx.shape

    Dx = diff_local(x, N, N)
    Dy = diff_local(y, N, N)

    Z = g2sSylvester(Dy, Dx, Zy, Zx, np.ones((m, 1)), np.ones((n, 1)))

    return Z


def g2s_weighted(x, y, Zx, Zy, Lxx, Lxy, Lyx, Lyy, N=3):
    """Purpose : Computes the Global Weighted Least Squares 
    reconstruction of a surface from its gradient field, whereby the 
    weighting is defined by a weighted Frobenius norm

    Use (syntax):
      Z = g2sWeighted( Zx, Zy, x, y, N, Lxx, Lxy, Lyx, Lyy )

    Input Parameters :
      Zx, Zy := Components of the discrete gradient field
      x, y := support vectors of nodes of the domain of the gradient
      N := number of points for derivative formulas (default=3)
      Lxx, Lxy, Lyx, Lyy := Each matrix Lij is the covariance matrix of 
      the gradient's i-component the in j-direction.

    Return Parameters :
      Z := The reconstructed surface

    Description and algorithms:
      The algorithm solves the normal equations of the Weighted Least 
      Squares cost function, formulated by matrix algebra:
      e(Z) = || Lyy^(-1/2) * (D_y * Z - Zy) * Lyx^(-1/2) ||_F^2 +
                  || Lxy^(-1/2) * ( Z * Dx' - Zx ) * Lxx^(-1/2) ||_F^2
      The normal equations are a rank deficient Sylvester equation which 
      is solved by means of Householder reflections and the 
      Bartels-Stewart algorithm.
    """

    if Zx.shape != Zy.shape:
        raise ValueError("Gradient components must be the same size")

    if (np.asmatrix(Zx).shape[1] != len(x) or
            np.asmatrix(Zx).shape[0] != len(y)):
        raise ValueError("Support vectors must have the same size as the "
                         "gradient")

    m, n = Zx.shape

    Dx = diff_local(x, N, N)
    Dy = diff_local(y, N, N)

    Wxx = la.sqrtm(Lxx)
    Wxy = la.sqrtm(Lxy)
    Wyx = la.sqrtm(Lyx)
    Wyy = la.sqrtm(Lyy)

    # Solution for Zw (written here Z)
    u = mldivide(Wxy, np.ones((m, 1)))
    v = mldivide(Wyx, np.ones((n, 1)))

    A = mldivide(Wyy, np.dot(Dy, Wxy))
    B = mldivide(Wxx, np.dot(Dx, Wyx))
    F = mldivide(Wyy, mrdivide(Zy, Wyx))
    G = mldivide(Wxy, mrdivide(Zx, Wxx))

    Z = g2sSylvester(A, B, F, G, u, v)

    # "Unweight" the solution
    Z = Wxy * Z * Wyx
    return Z


def g2s_spectral(x, y, Zx, Zy, mask, N=3):
    """Purpose : Computes the Global Least Squares reconstruction of a 
    surface from its gradient field with Spectral filtering, using either
    polynomial, cosine, or Fourier bases.

    Use (syntax):
      Z = g2sSpectral( Zx, Zy, x, y, N, Mask, basisFns )
      Z = g2sSpectral( Zx, Zy, x, y, N, Mask )

    Input Parameters :
      Zx, Zy := Components of the discrete gradient field
      x, y := support vectors of nodes of the domain of the gradient
      N := number of points for derivative formulas (default=3)
      Mask := either a 1x2 matrix, [p,q], specifying the size of a low 
      pass filter, or a general mxn spectral mask.
      basisFns := 'poly', 'cosine', 'fourier', specifying the type of 
      basis functions used for regularization.  Defaults is polynomial.  
      For arbitrary node spacing, only polnomial filtering is 
      implemented.

    Return Parameters :
      Z := The reconstructed surface

    Description and algorithms:
      The algorithm solves the normal equations of the Least Squares 
      cost function, formulated by matrix algebra:
      e(C) = || D_y * By * C * Bx' - Zy ||_F^2 + || By * C * Bx' * Dx' 
             - Zx ||_F^2
      The surface is parameterized by its spectrum, C, w.r.t. sets of 
      orthogonal basis functions, as,
      Z = By * C * Bx'
      The normal equations are a rank deficient Sylvester equation which 
      is solved by means of Householder reflections and the 
      Bartels-Stewart algorithm.
    """

    ## For now, use only the polynomial basis function.
    ## The mask implementation is basic, too.

    if Zx.shape != Zy.shape:
        raise ValueError("Gradient components must be the same size")

    if (np.asmatrix(Zx).shape[1] != len(x) or
            np.asmatrix(Zx).shape[0] != len(y)):
        raise ValueError("Support vectors must have the same size as the "
                         "gradient")

    # Low Pass Filter
    if mask.shape == [1, 2]:
        p = mask[0, :]
        q = mask[1, :]
    # Arbitrary Mask
    elif mask.shape == Zx.shape:
        p, q = Zx.shape

    Dx = diff_local(x, N, N)
    Dy = diff_local(y, N, N)

    # Generate the Basis Functions
    Bx = np.array(dop(x)[0])
    By = np.array(dop(y)[0])

    # Solve the Sylvester Equation
    A = np.dot(Dy, By)
    B = np.dot(Dx, Bx)
    F = np.dot(Zy, Bx)
    G = np.dot(By.T, Zx)

    C = np.zeros((p, q))
    C[0, 1:] = mrdivide(G[0, :], B[:, 1:].T)
    C[1:, 0] = mldivide(A[:, 1:p], F[:, 0])
    C[1:, 1:] = solve_sylvester(np.dot(A[:, 1:p].T, A[:, 1:p]),
                                np.dot(B[:, 1:q].T, B[:, 1:q]),
                                np.dot(-A[:, 1:p].T, F[:, 1:])
                                - np.dot(G[1:, :], B[:, 1:q]))

    # Apply the Spectral Filter (if necessary)
    if mask.shape != [1, 2]:
        C *= mask

    # Taking the real part is only neccessary for the Fourier basis:
    # Z = real( By * C * Bx' ) ; % Z = By * C * Bx' for all others
    Z = np.dot(By, np.dot(C, Bx.T))

    return Z


def g2s_dirichlet(x, y, Zx, Zy, ZB=np.array([]), N=3):
    """Purpose : Computes the Global Least Squares reconstruction of a 
    surface from its gradient field with Dirichlet Boundary conditions.  
    The solution surface is thereby constrained to have fixed values at 
    the boundary, specified by ZB.

    Use (syntax):
      Z = g2sDirichlet( Zx, Zy, x, y )
      Z = g2sDirichlet( Zx, Zy, x, y, N )
      Z = g2sDirichlet( Zx, Zy, x, y, N, ZB )

    Input Parameters :
      Zx, Zy := Components of the discrete gradient field
      x, y := support vectors of nodes of the domain of the gradient
      N := number of points for derivative formulas (default=3)
      ZB := a matrix specifying the value of the solution surface at the
         boundary ( omitting this assumes ZB=zeros(m,n) )

    Return Parameters :
      Z := The reconstructed surface

    Description and algorithms:
      The algorithm solves the normal equations of the Least Squares 
      cost function, formulated by matrix algebra:
      e(Z) = || D_y * Z - Zy ||_F^2 + || Z * Dx' - Zx ||_F^2
      The normal equations are a rank deficient Sylvester equation which 
      is solved by means of Householder reflections and the 
      Bartels-Stewart algorithm.
    """

    if Zx.shape != Zy.shape:
        raise ValueError("Gradient components must be the same size")

    if (np.asmatrix(Zx).shape[1] != len(x) or
            np.asmatrix(Zx).shape[0] != len(y)):
        raise ValueError("Support vectors must have the same size as the "
                         "gradient")

    m, n = Zx.shape

    Dx = diff_local(x, N, N)
    Dy = diff_local(y, N, N)

    # Set Z equal to ZB for memory useage (avoids using ZI)
    if len(ZB) != 0:
        Z = ZB
    else:
        Z = np.zeros((m, n))

    return Z
