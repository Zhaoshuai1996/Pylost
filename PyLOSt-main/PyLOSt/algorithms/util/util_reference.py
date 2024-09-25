# coding=utf-8
import numpy as np
from scipy.sparse.coo import coo_matrix

from PyLOSt.algorithms.util.util_fit import getXYGrid
from PyLOSt.algorithms.util.util_math import nbTermsPoly
from PyLOSt.algorithms.util.util_poly import legendre_xy, zernike_xy


# import numba as nb
#
# @nb.jit(nopython=False)#(parallel=True)
# def get_ref_nb(self, xx_r, yy_r, sdiff, slc_i, slc_j):
#     index_mat = np.arange(xx_r.size).reshape(xx_r.shape)
#     nbTerms = xx_r.size
#     sz = 0
#     for val in sdiff:
#         mask = ~np.isnan(val)
#         sz = sz + mask.nonzero()[0].size
#     rows = np.empty((sz, 2), dtype=np.int32)  # counts overlaps x valid pixels in each ovrlap
#     cols = np.empty((sz, 2), dtype=np.int32)  # counts number of terms to fit in the reference
#     data = np.empty((sz, 2), dtype=np.int32)
#     row = 0
#     for i, val in enumerate(sdiff):
#         if self.verbose >= 1: print('Building ref matrix : {}'.format(i))
#         mask = ~np.isnan(val)
#         osz = mask.nonzero()[0].size
#
#         if self.ref_extract_type == 'full':
#             rows[row:row+osz, :] = np.arange(row, row+osz).reshape(-1, 1)
#             im = index_mat[slc_i[i][1:3]]
#             cols[row:row+osz, 0] = im.ravel()[mask.ravel()]
#             jm = index_mat[slc_j[i][1:3]]
#             cols[row:row+osz, 1] = jm.ravel()[mask.ravel()]
#             data[row:row+osz, :] = np.array([-1, 1]).reshape(1, -1)
#             row = row + osz
#         # else:
#         #     xi = xx_r[slc_i[key][1:]][mask]
#         #     yi = yy_r[slc_i[key][1:]][mask]
#         #     xj = xx_r[slc_j[key][1:]][mask]
#         #     yj = yy_r[slc_j[key][1:]][mask]
#         #     Ti = buildCforExtractRef(self, xi, yi)  # osz x nbTerms
#         #     Tj = buildCforExtractRef(self, xj, yj)  # osz x nbTerms
#         #     nbTerms = Ti.shape[-1]  # has to be constant across overlaps
#         #
#         #     for l in range(0, osz):
#         #         rows += [row] * nbTerms
#         #         row += 1
#         #     cols += list(col_offset + np.arange(nbTerms)) * osz
#         #     data += list((Tj - Ti).ravel())
#     shape = (row, nbTerms)
#     return data, rows, cols, shape
#
# def get_A_Ref_numba(self, sarr, sdiff, mX, mY, pix_size, slc_i, slc_j, otype, col_offset=0, scaleXY=True, filter_rows=[], prog_block=0):
#     xx_r, yy_r = getXYGrid(np.ones_like(sarr[0], dtype='float'), pix_size=pix_size, order=2)
#     xscale = max(xx_r.ravel()) if scaleXY else 1
#     yscale = max(yy_r.ravel()) if scaleXY else 1
#     xx_r = xx_r / xscale
#     yy_r = yy_r / yscale
#     data, rows, cols, shape = get_ref_nb(self, xx_r, yy_r, tuple(sdiff.values()), tuple(slc_i.values()), tuple(slc_j.values()))
#     B_coo = coo_matrix((data.ravel(), (rows.ravel(), cols.ravel())), shape)
#     self.increment_progress(prog_block * 0.5)
#     return B_coo, [xscale, yscale]

def get_A_Ref(self, sarr, sdiff, mX, mY, pix_size, slc_i, slc_j, otype, col_offset=0, scaleXY=True, filter_rows=[],
              prog_block=0):
    xx_r, yy_r = getXYGrid(np.ones_like(sarr[0], dtype='float'), pix_size=pix_size, order=2)
    xscale = max(xx_r.ravel()) if scaleXY else 1
    yscale = max(yy_r.ravel()) if scaleXY else 1
    xx_r = xx_r / xscale
    yy_r = yy_r / yscale
    index_mat = np.arange(xx_r.size).reshape(xx_r.shape)
    sz = 0
    for val in sdiff.values():
        mask = ~np.isnan(val)
        sz = sz + mask.nonzero()[0].size  # TODO: why not numpy.count_nonzero
    rows = np.empty((sz, 2), dtype=np.int32)  # counts overlaps x valid pixels in each ovrlap
    cols = np.empty((sz, 2), dtype=np.int32)  # counts number of terms to fit in the reference
    data = np.empty((sz, 2), dtype=np.int32)
    row = 0
    new_block = prog_block * 0.5 * 1 / len(sdiff)
    for k, key in enumerate(sdiff):
        if self.verbose >= 1:
            print('Building ref matrix : {}'.format(key))
        mask = ~np.isnan(sdiff[key])
        osz = sdiff[key][mask].size

        if self.ref_extract_type == 'full':
            nbTerms = xx_r.size
            rows[row:row + osz, :] = np.arange(row, row + osz).reshape(-1, 1)
            cols[row:row + osz, 0] = index_mat[slc_i[key][1:]][mask]
            cols[row:row + osz, 1] = index_mat[slc_j[key][1:]][mask]
            data[row:row + osz, :] = np.array([-1, 1]).reshape(1, -1)
            row = row + osz
        else:
            xi = xx_r[slc_i[key][1:]][mask]
            yi = yy_r[slc_i[key][1:]][mask]
            xj = xx_r[slc_j[key][1:]][mask]
            yj = yy_r[slc_j[key][1:]][mask]
            Ti = buildCforExtractRef(self, xi, yi)  # osz x nbTerms
            Tj = buildCforExtractRef(self, xj, yj)  # osz x nbTerms
            nbTerms = Ti.shape[-1]  # has to be constant across overlaps

            for l in range(0, osz):
                rows += [row] * nbTerms
                row += 1
            cols += list(col_offset + np.arange(nbTerms)) * osz
            data += list((Tj - Ti).ravel())
        self.increment_progress(new_block)

    shape = (row, nbTerms)
    B_coo = coo_matrix((data.ravel(), (rows.ravel(), cols.ravel())), shape)
    self.increment_progress(prog_block * 0.5)
    return B_coo, [xscale, yscale]


def buildCforExtractRef(self, xv, yv):
    if self.ref_extract_type == 'poly':
        return buildPoly(self, xv, yv)
    elif self.ref_extract_type == 'legendre':
        return buildLegendre(self, xv, yv)
    elif self.ref_extract_type == 'zernike':
        return buildZernike(self, xv, yv)


def buildZernike(self, xv, yv):
    Z, _, _ = zernike_xy(xv, yv, degree=self.end_deg, start_degree=self.start_deg)
    return Z


def buildLegendre(self, xv, yv):
    # nbStartTerms = nbTermsLegendre2D(stDeg - 1)
    # Z = legvander2d(xv, yv, [enDeg, enDeg])
    # if nbStartTerms > 0: Z[:, 0:nbStartTerms] = 0
    Z, _, _ = legendre_xy(xv, yv, degree=self.end_deg, start_degree=self.start_deg)
    return Z


def buildPoly(self, xv, yv):
    """
    Expand index matrix to extract reference from measured data

    :param a: IndexMatrix i-th overlap row
    :param a_i: Pixelwise IndexMatrix i-th overlap rows
    :param xv: X-grid data in reference coordinates
    :param yv: Y-grid data in reference coordinates
    :return: Polynomial term matrix with shape (size of subap valid pixels, num of terms)
    """
    enDeg = self.end_deg
    stDeg = self.start_deg
    nbTerms = nbTermsPoly(stDeg - 1, enDeg)
    C = np.full((xv.size, nbTerms), np.nan)
    i = 0
    for n in range(stDeg, enDeg + 1):
        for k in range(0, n + 1):
            C[:, i] = xv ** (k) * yv ** (n - k)
            i = i + 1
    return C
