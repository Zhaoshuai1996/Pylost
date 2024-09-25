# coding=utf-8
"""
Created on Mar 9, 2019

Util functions used in fast stitching algorithms

@author: ADAPA
"""
import cProfile
import pstats
import timeit

import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow, plot
from scipy import optimize
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from PyLOSt.algorithms.util.util_fit import fit2D, getXYGrid
from PyLOSt.algorithms.util.util_math import rms


def start_profiler():
    profiler = cProfile.Profile()
    profiler.enable()
    return profiler


def print_profiler(profiler):
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()


def flatten_list(a):
    return [item for sublist in a for item in sublist]


def filtBadPixels(self, sarr, pix_sz, key):
    """
    Filter pixels above n-std from global shape.

    :param self: Stitching function reference object
    :param sarr: Input data
    :param pix_sz: Pixel size
    :return: Filter mask
    """
    maxStd = float(self.filt_bad_pix_threshold)
    sz = sarr.shape
    mask = np.full(sarr.shape, False)
    deg = 2 if key == 'height' else 1
    for j in np.arange(0, sz[0]):
        _, resd, _ = fit2D(sarr[j], pix_size=pix_sz, degree=deg, retResd=True)
        rmsResd = rms(resd)
        mask[j, :, :] = resd < maxStd * rmsResd
    return mask


def optimize_correctors(self, A, E, P0, method, show_cost=False):
    P = P0
    self.show_cost = show_cost
    E = np.array(E)
    if E.ndim == 1:
        E = E[:, np.newaxis]
    constraint = ()
    bounds = None
    if self.ref_extract_type == 'full':
        if self.constraint_zero_sum:
            constraint = optimize.LinearConstraint(np.ones_like(P0), lb=0, ub=0)
        if self.bound_nstd > 0:
            bounds = [(x - self.bound_nstd * self.subap_avg_rms, x + self.bound_nstd * self.subap_avg_rms) for x in P0]

    start = timeit.default_timer()
    if method == 'least_squares_nonlinear':
        res = optimize.least_squares(errfunc, P0, args=(self, A, E))
        P = res.x
    elif method == 'least_squares_linear':
        if bounds is not None:
            bounds = (-self.bound_nstd * self.subap_avg_rms, self.bound_nstd * self.subap_avg_rms)
            res = optimize.lsq_linear(A, E.ravel(), bounds=bounds)
        else:
            res = optimize.lsq_linear(A, E.ravel())
        P = res.x
    elif method == 'minimize':
        res = optimize.minimize(costfunc, P0, args=(self, A, E), constraints=constraint, bounds=bounds)
        P = res.x
    else:
        P = P0
        res = None
    # if self.verbose >= 1:
    #     print('Result : ', res)
    stop = timeit.default_timer()
    if self.verbose >= 1:
        print('Time: ', stop - start)
    return P


def errfunc(P, self, A, E):
    AP = A.dot(P)
    AP = AP[:, np.newaxis]
    err_F = AP - E
    # if self.verbose >= 2:
    #     print('rms(err_F): {:.5f}'.format(rms(err_F)))
    if self.show_cost:
        self.set_info(np.nansum(err_F ** 2), 'costfunc')
    return err_F[~np.isnan(err_F)]


def costfunc(P, self, A, E):
    AP = A.dot(P)
    AP = AP[:, np.newaxis]
    err_F = AP - E
    if self.verbose >= 2:
        print('rms(costfunc): {:.5f}'.format(rms(err_F)))
    if self.show_cost:
        self.set_info(np.nansum(err_F ** 2), 'costfunc')
    return np.nansum(err_F ** 2)


def get_sparse_inv(A, E):
    A = A.asfptype()
    u, s, vt = scipy.sparse.linalg.svds(A, k=np.min(A.shape) - 1)
    invA = np.matmul(vt.T, np.matmul(np.diag(1 / s), u.T))
    return np.matmul(invA, E)


def calc_inverse(self, A, E):
    if A.shape[0] == A.shape[1]:  # square
        return spsolve(A, E)
    else:
        try:
            return spsolve(A.T * A, A.T * E)
        except Exception as e:
            print(e)
            if self.inv_type == 'SVD':
                return get_sparse_inv(A, E)
            else:
                invA = np.linalg.pinv(A.toarray())
                return np.matmul(invA, E)
    # return None


def get_A_allpix(self, A, sdiff, mX, mY, pix_size, res_item, otype, prog_block=0):
    """

    :param A: IndexMatrix
    :param sdiff: Overlap errors array of all pixels
    :param oXArr: Translation offset array
    :return: Correctors array
    """
    mask_mir = np.full(res_item.shape, True, dtype=bool)
    xx_mir, yy_mir = getXYGrid(mask_mir, pix_size=pix_size, order=2, mask=mask_mir)

    nbPat = len(mX)
    rows = []  # counts overlaps x valid pixels in each ovrlap
    cols = []  # counts number of subapertures x number of fit params (e.g. 3 for piston/pitch/roll)
    data = []
    row = 0
    new_block = prog_block * 0.5 * 1 / len(A)
    for k, key in enumerate(A):
        # if self.verbose >= 1:
        #     print('Building full matrix : {}'.format(key))
        i, j = key
        mask = ~np.isnan(sdiff[key])
        osz = sdiff[key][mask].size

        ox = max(mX[i], mX[j])
        oy = max(mY[i], mY[j])
        slc = (slice(oy, oy + mask.shape[-2]), slice(ox, ox + mask.shape[-1]))
        xm = xx_mir[slc][mask]
        ym = yy_mir[slc][mask]

        for l in range(0, osz):
            rows += [row] * (2 if otype in ['slopes_x', 'slopes_y'] else 6)
            row += 1
        if otype == 'height':
            cols += (list(key) + [x + nbPat for x in key] + [x + 2 * nbPat for x in key]) * osz
        elif otype == 'slopes_y':
            cols += [x + nbPat for x in key] * osz
        elif otype == 'slopes_x':
            cols += [x + 2 * nbPat for x in key] * osz
        data += list(A[key]) * osz if otype in ['slopes_x', 'slopes_y'] else flatten_list(
            [(list(A[key]) + [x * ym[p] for x in A[key]] + [x * xm[p] for x in A[key]]) for p in range(osz)])
        self.increment_progress(new_block)
    shape = (row, 3 * nbPat)  # piston/pitch/roll
    A_coo = coo_matrix((data, (rows, cols)), shape)
    self.increment_progress(prog_block * 0.5)
    return A_coo


def get_sparse_from_dict(A, shape, mode=1):
    rows = []
    cols = []
    data = []
    no, np = shape
    shape_coo = (3 * no, 3 * np) if mode == 2 else shape
    for i, key in enumerate(A):
        if mode == 1:  # Correctors as overlaps x 3 matrix
            rows += [i, i]
            cols += list(key)
            data += list(A[key])
        elif mode == 2:  # Correctors as overlaps * 3 array
            rows += [i, i] + [i + no, i + no] + [i + 2 * no, i + 2 * no]
            cols += list(key) + [key[0] + np, key[1] + np] + [key[0] + 2 * np, key[1] + 2 * np]
            data += list(A[key]) * 3

    return coo_matrix((data, (rows, cols)), shape_coo)


def getOverlaps2D(self, oXArr, oYArr, data, otype, pix_sz, cor_terms=[True, True, True], cor_ref=False, fit_data=True,
                  showPlots=False, prog_block=0):
    """
    Build overlap errors

    :param self: Stitching function reference object
    :param oXArr: Translation offset array
    :param data: Subaperture array
    :param otype: Data type slope/height
    :param cor_terms: Correction terms - piston/roll/pitch
    :param extractRef: Flag to retrieve reference from fitting polynomial
    :param degRef: Polynomial degree to use for extracting reference from fitting
    :param startDegRef: Start degree for reference. Usually degree<2 are excluded as theses reference orders cannot be extracted from measured data
    :return: IndexMatrix, overlaps matrix, overalap errors pitch/roll/piston, max overlap count, valid overlaps, valid subapertures, subaperture array, maximum overlap order, piston array
    """
    nPat = len(oXArr)
    validPatches = np.full([nPat], True)

    if otype in ['slopes_x', 'slopes_y', 'height']:
        sarr = data

        if self.num_exclude_subaps > 0:
            sarr[:self.num_exclude_subaps, :, :] = np.nan
            sarr[-self.num_exclude_subaps:, :, :] = np.nan

        sarr_mask = np.isnan(sarr)
        ## Filter: bad pixels above n*std from ideal shape
        if self.filt_bad_pix:
            maskBadPix = filtBadPixels(self, sarr, pix_sz, otype)
            sarr_mask = sarr_mask * maskBadPix
            sarr[~sarr_mask] = np.nan

        res_mask_szY = sarr.shape[-2] + max(oYArr)
        res_mask_szX = sarr.shape[-1] + max(oXArr)
        mask_mir = np.full([res_mask_szY, res_mask_szX], True)
        xx_mir, yy_mir = getXYGrid(mask_mir, pix_size=pix_sz, order=2, mask=mask_mir)
        if self.verbose >= 1:
            print('Fin prepare data objects')

        if self.remove_outlier_subapertures:
            # mask of validpatches: exclude outlier patches
            sarr_rms = np.sqrt(np.nanmean(np.nanmean((sarr) ** 2, axis=2), axis=1))
            med_savg = np.median(sarr_rms[validPatches])
            Q = np.abs(sarr_rms - med_savg) < 4 * np.std(
                sarr_rms[validPatches] - med_savg)  # TODO: use proximity to neighbours instead
            sarr[~Q, :, :] = np.nan
            sarr_mask[~Q, :, :] = False
            validPatches = validPatches * Q
            if self.verbose >= 1:
                print('Fin remove outlier subapertures')

        if showPlots:
            plotSt(sarr, oxArr=oXArr, pnum=4, title='Data after pre processing')

        sz = sarr.shape
        slc_i = {}
        slc_j = {}
        sdiff = {}
        A = {}
        E = []
        E_resd = []
        self.increment_progress(prog_block * 0.1)
        cnt = nPat * (nPat - 1) / 2
        new_block = prog_block * 0.9 * 1 / cnt
        for i, oxi in enumerate(oXArr):
            oyi = oYArr[i]
            for j, oxj in enumerate(oXArr):
                if j <= i:
                    continue  # we already looped over these overlaps
                if not validPatches[i]:
                    continue
                if not validPatches[j]:
                    continue

                oyj = oYArr[j]
                dx = oxj - oxi
                dy = oyj - oyi
                if (sz[-1] - np.abs(dx)) * (sz[-2] - np.abs(dy)) < self.min_overlap * sz[-1] * sz[-2]:
                    continue  # not enough overlap (first approximation)

                slc_i[(i, j)] = (
                    i, slice(dy, None) if dy >= 0 else slice(None, dy), slice(dx, None) if dx >= 0 else slice(None, dx))
                slc_j[(i, j)] = (
                    j, slice(None, -dy) if dy > 0 else slice(-dy, None),
                    slice(None, -dx) if dx > 0 else slice(-dx, None))
                diff = sarr[slc_i[(i, j)]] - sarr[slc_j[(i, j)]]
                if np.sum(~np.isnan(diff.ravel())) < self.min_overlap * np.min(
                        [np.sum(~np.isnan(sarr[i].ravel())), np.sum(~np.isnan(sarr[j].ravel()))]):
                    continue  # not enough overlap (second approximation excluding NaNs)
                if np.isnan(np.nanmean(diff.ravel())):
                    continue

                sdiff[(i, j)] = diff
                A[(i, j)] = [-1, 1]
                filt_terms = np.array([0] * 7)
                filt_terms[0:3] = cor_terms
                if otype == 'height' and fit_data:
                    ox = max(oxi, oxj)
                    oy = max(oyi, oyj)
                    slc_mir = (slice(oy, oy + diff.shape[-2]), slice(ox, ox + diff.shape[-1]))
                    if np.any(filt_terms):
                        fo, resd, _ = fit2D(diff, pix_size=pix_sz, filter_terms_poly=filt_terms,
                                            retResd=cor_ref, xv=xx_mir[slc_mir], yv=yy_mir[slc_mir])
                    else:
                        fo = [0, 0, 0]
                        resd = diff
                    E.append(list(fo[:3]))
                    if resd is not None:
                        E_resd += list(resd.ravel())
                elif otype == 'slopes_x':
                    val = np.nanmean(diff.ravel()) if self.cor_pitch else 0
                    E.append([0, 0, val])
                    E_resd += list(diff.ravel() - val)
                elif otype == 'slopes_y':
                    val = np.nanmean(diff.ravel()) if self.cor_roll else 0
                    E.append([0, val, 0])
                    E_resd += list(diff.ravel() - val)
                self.increment_progress(new_block)

        if len(A) < cnt:
            self.increment_progress(prog_block * 0.9 * (cnt - len(A)) / cnt)

        return sdiff, A, E, E_resd, validPatches, slc_i, slc_j


############################################
# Correct and join
def correctAndJoin(self, C, sarr, pix_size, res_item, res_intensity, otype, mX, mY, C_ref=None, ref_scale=[1, 1],
                   prog_block=0):
    """
    Apply corrections and join the subapertures

    :param c: Correctors
    :param sarr: Subaperture array
    :param otype: Data type - slope/height
    :param oXArr: Translation offset array
    :return: Stitched image
    """
    scan_item_cor = np.full(sarr.shape, np.nan, dtype=sarr.dtype)
    mask_mir = np.full(res_item.shape, True, dtype=bool)
    xx, yy = getXYGrid(mask_mir, pix_size=pix_size, order=2, mask=mask_mir)
    nPat, ny, nx = sarr.shape
    ref_ext = None

    f_ref = 0.3
    f = 1 - (f_ref if self.cor_reference_extract else 0)
    corRef = 0
    if self.cor_reference_extract:
        xvRef, yvRef = getXYGrid(np.ones_like(sarr[0], dtype='float'), pix_size=pix_size, order=2)
        if self.ref_extract_type == 'full':
            ref_ext = corRef = np.reshape(C_ref, xvRef.shape)
        else:
            ref_ext = corRef = getCorrectionReference(self, C_ref, xvRef / ref_scale[0], yvRef / ref_scale[1])
        self.increment_progress(prog_block * f_ref)
    new_block = prog_block * f * 1 / len(sarr)
    for j, s in enumerate(sarr):
        ox = mX[j]
        oy = mY[j]
        slc = (slice(oy, oy + ny), slice(ox, ox + nx))
        cor = getCorrectionMap(C[j, :], otype, xx[slc], yy[slc])
        temp = sarr[j, :, :] + cor + corRef
        if self.use_threshold == 'post_process':
            temp = self.maskByThreshold(temp, sarr[j, :, :])
        res_item[slc] = np.nansum(np.dstack((res_item[slc], temp)), 2)
        res_intensity[slc] = np.nansum([res_intensity[slc], (~np.isnan(temp)).astype(int)], axis=0)
        scan_item_cor[j] = temp
        if self.showPlots:
            data = np.divide(res_item, res_intensity)
            plotSt(data, pnum=9, title='Stitched profile : ' + str(j), pauseTime=0.1)
        self.increment_progress(new_block)

    sf = np.divide(res_item, res_intensity)
    err_val = self.get_algorithm_error(mX, mY, scan_item_cor, sf)
    return err_val, sf, scan_item_cor, ref_ext


def getCorrectionMap(c, otype, xx_j, yy_j):
    """
    Get correcion map for j'th subaperture

    :param c: jth corrector
    :param xx_j: X-grid postions
    :param yy_j: Y-grid postions
    :return:
    """
    if otype in ['slopes_x', 'slopes_y']:
        return c[1] + c[2]
    elif otype == 'height':
        return c[0] + c[1] * yy_j + c[2] * xx_j
    else:
        return 0


##########################################
## Correction reference
from numpy.polynomial.legendre import legval2d
from PyLOSt.algorithms.util.util_poly import zernike_xy


def getCorrectionReference(self, c_ref, xx, yy):
    if self.ref_extract_type == 'poly':
        return getCorrectionReferencePoly(self, c_ref, xx, yy)
    elif self.ref_extract_type == 'legendre':
        return getCorrectionReferenceLegendre(self, c_ref, xx, yy)
    elif self.ref_extract_type == 'zernike':
        return getCorrectionReferenceZernike(self, c_ref, xx, yy)


def getCorrectionReferenceZernike(self, c_ref, xx, yy):
    enDeg = self.end_deg
    _, _, corRef = zernike_xy(xx, yy, degree=enDeg, coef=c_ref)
    return corRef


def getCorrectionReferenceLegendre(self, c_ref, xx, yy):
    # nbStartTerms = nbTermsLegendre2D(startDeg-1)
    # if nbStartTerms > 0: c_ref[0:nbStartTerms] = 0
    corRef = legval2d(xx, yy, c_ref)
    return corRef


def getCorrectionReferencePoly(self, c_ref, xx, yy):
    """

    :param c_ref: reference coefficients♦
    :param xx: reference x grid
    :param yy: reference y grid
    :return: reference map
    """
    corRef = 0
    enDeg = self.end_deg
    stDeg = self.start_deg
    i = 0
    for n in range(stDeg, enDeg + 1):
        for k in range(0, n + 1):
            corRef = corRef + c_ref[i] * xx ** (k) * yy ** (n - k)
            i += 1
    return corRef


def plotSt(a3, oxArr=[], pnum=1, title='tilte', pauseTime=0, alpha=0.3):
    try:
        szPat = a3.shape
        fig = plt.figure(pnum)
        plt.title(title)
        if a3.ndim == 3:
            if np.any(oxArr):
                res_szX = szPat[-1] + max(oxArr)
                res_szY = szPat[-2]
                axes = fig.gca()
                axes.set_xlim([0, res_szX])
                # hold(True)
                for i, ox in enumerate(oxArr):
                    imshow(a3[i], extent=[ox, ox + szPat[-1], 0, szPat[-2]], alpha=alpha)
                    plt.pause(0.1 if pauseTime <= 0 else pauseTime)
            else:
                for a in a3:
                    imshow(a)
                    plt.pause(0.1 if pauseTime <= 0 else pauseTime)
        elif a3.ndim == 2:
            imshow(a3)
        elif a3.ndim == 1:
            plot(a3)
        # plt.show()
        if pauseTime > 0:
            plt.pause(pauseTime)
            # plt.close()
    except Exception as e:
        print('plotSt <- util_functions')
        print(e)


def binData(arr, binX, binY):
    """
    Bin data

    :param arr: Input data
    :param binX: Bin size along X
    :param binY: Bin size along Y
    :return: Binned data
    """
    retArr = np.array([], dtype=arr.dtype)
    if arr.ndim == 3:
        for j in np.arange(0, arr.shape[0]):
            dj = bin2DData(arr[j], binX, binY)
            sdj = dj.shape
            retArr = np.append(retArr, dj.reshape(1, sdj[0], sdj[1]), axis=0) if np.any(retArr) else dj.reshape(1,
                                                                                                                sdj[0],
                                                                                                                sdj[1])
    elif arr.ndim == 2:
        retArr = bin2DData(arr, binX, binY)
    return retArr


def bin2DData(data, binX, binY):
    """
    Bin 2D data

    :param data:
    :param binX:
    :param binY:
    :return:
    """
    sz = data.shape
    m_bins = int(sz[0] / binY)
    n_bins = int(sz[1] / binX)
    data = data[:m_bins * binY, :n_bins * binX]
    return np.nanmean(np.nanmean(data.reshape(m_bins, binY, n_bins, binX), 3), 1)
