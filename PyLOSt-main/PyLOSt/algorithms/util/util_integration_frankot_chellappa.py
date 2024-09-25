# coding=utf-8
import numpy as np

from PyLOSt.util.commons import alertMsg


def frankot_chellappa(dzdx, dzdy, reflec_pad=True):
    """Python version of FRANKOTCHELLAPPA Matlab code:
    http://www.peterkovesi.com/matlabfns/Shapelet/frankot_chellappa.m

    Copyright notice for original Matlab code:
    -----------------------------------------------------------------------
    FRANKOTCHELLAPPA  - Generates integrable surface from gradients

    An implementation of Frankot and Chellappa'a algorithm for
    constructing an integrable surface from gradient information.

    Usage:      z = frankot_chellappa(dzdx,dzdy)

    Arguments:  dzdx,  - 2D matrices specifying a grid of gradients of z
                dzdy     with respect to x and y.

    Returns:    z      - Inferred surface heights.

    Reference:

    Robert T. Frankot and Rama Chellappa
    A Method for Enforcing Integrability in Shape from Shading
    IEEE PAMI Vol 10, No 4 July 1988. pp 439-451

    Note this code just implements the surface integration component of
    the paper (Equation 21 in the paper).  It does not implement their
    shape from shading algorithm.

    Copyright (c) 2004 Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    http://www.csse.uwa.edu.au/

    Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation files
    (the "Software"), to deal in the Software without restriction,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    The Software is provided "as is", without warranty of any kind.

    October 2004

    Python version written by Ruxandra Cojocaru, July 2017

    Modified frequency domain to correspond to g2s grid, added padding
    and added factors to equation and reconstruction
    """

    if not bool(dzdx.shape == dzdy.shape):
        alertMsg('frankot_chellappa', 'size of gradient matrices must match')
        return

    if reflec_pad:
        dzdx, dzdy = _reflec_pad_grad_fields(dzdx, dzdy)

    (rows, cols) = dzdx.shape

    # The following sets up matrices specifying frequencies in the x and
    # y directions corresponding to the Fourier transforms of the
    # gradient data.  They range from -0.5 cycles/pixel to
    # + 0.5 cycles/pixel. The fiddly bits in the line below give the
    # appropriate result depending on whether there are an even or odd
    # number of rows and columns

    (wx, wy) = np.meshgrid(np.pi * (np.arange(1, cols + 1)
                                    - (np.fix(cols / 2.0) + 1))
                           / (cols - np.mod(cols, 2)),
                           np.pi * (np.arange(1, rows + 1)
                                    - (np.fix(rows / 2.0) + 1))
                           / (rows - np.mod(rows, 2)))

    # Quadrant shift to put zero frequency at the appropriate edge
    wx = np.fft.ifftshift(wx)
    wy = np.fft.ifftshift(wy)

    # Fourier transforms of gradients
    Fdzdx = np.fft.fft2(dzdx)
    Fdzdy = np.fft.fft2(dzdy)

    # Integrate in the frequency domain by phase shifting by pi/2 and
    # weighting the Fourier coefficients by their frequencies in x and y
    # and then dividing by the squared frequency.  eps is added to the
    # denominator to avoid division by 0.

    # Equation 21 from the Frankot & Chellappa paper
    # ADDED A * (-1)
    Z = (-1 * (-1j * wx * Fdzdx - 1j * wy * Fdzdy)
         / (wx ** 2 + wy ** 2 + np.spacing(1)))

    # Reconstruction
    rec = np.real(np.fft.ifft2(Z))

    # Source:
    # http://www.cs.cmu.edu/~ILIM/projects/IM/aagrawal/software.html
    rec = rec / 2.0

    # Source:
    # https://github.com/wavepy/wavepy/blob/master/wavepy/surface_from_grad.py
    if reflec_pad:
        return _one_forth_of_array(rec)
    else:
        return rec

    # return rec


def _one_forth_of_array(array):
    """
    Undo for the function
    :py:func:`wavepy:surface_from_grad:_reflec_pad_grad_fields`
    """

    array, _ = np.array_split(array, 2, axis=0)

    return np.array_split(array, 2, axis=1)[0]


def _reflec_pad_grad_fields(del_func_x, del_func_y):
    """Source:
    https://github.com/wavepy/wavepy/blob/master/wavepy/surface_from_grad.py

    Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.

    Copyright 2015. UChicago Argonne, LLC. This software was produced
    under U.S. Government contract DE-AC02-06CH11357 for Argonne
    National Laboratory (ANL), which is operated by UChicago Argonne,
    LLC for the U.S. Department of Energy. The U.S. Government has
    rights to use, reproduce, and distribute this software.  NEITHER THE
    GOVERNMENT NOR UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR
    IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If
    software is modified to produce derivative works, such modified
    software should be clearly marked, so as not to confuse it with the
    version available from ANL.

    Additionally, redistribution and use in source and binary forms,
    with or without modification, are permitted provided that the
    following conditions are met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials
          provided with the distribution.

        * Neither the name of UChicago Argonne, LLC, Argonne National
          Laboratory, ANL, the U.S. Government, nor the names of its
          contributors may be used to endorse or promote products
          derived from this software without specific prior written
          permission.

    THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
    Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    This fucntion pads the gradient field in order to obtain a
    2-dimensional reflected function. The idea is that, by having an
    reflected function, we avoid discontinuity at the edges.
    This was inspired by the code of the function DfGBox, available in
    theMATLAB File Exchange website:
    https://www.mathworks.com/matlabcentral/fileexchange/45269-dfgbox
    """

    del_func_x_c1 = np.concatenate((del_func_x,
                                    del_func_x[::-1, :]), axis=0)

    del_func_x_c2 = np.concatenate((-del_func_x[:, ::-1],
                                    -del_func_x[::-1, ::-1]), axis=0)

    del_func_x = np.concatenate((del_func_x_c1, del_func_x_c2), axis=1)

    del_func_y_c1 = np.concatenate((del_func_y,
                                    -del_func_y[::-1, :]), axis=0)

    del_func_y_c2 = np.concatenate((del_func_y[:, ::-1],
                                    -del_func_y[::-1, ::-1]), axis=0)

    del_func_y = np.concatenate((del_func_y_c1, del_func_y_c2), axis=1)

    return del_func_x, del_func_y
