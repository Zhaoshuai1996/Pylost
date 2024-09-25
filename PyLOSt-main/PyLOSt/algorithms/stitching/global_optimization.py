# coding=utf-8
'''
Created on Mar 9, 2019

Global optimization method uses all the overlaps above a threshold and creates an optimization function of overlap errors.
In the SIMPLE mode the overlap errors are averaged over all pixels for each overlap, and in FULL mode all the pixels are used in optimization function

@author: ADAPA
'''

import numpy as np
from scipy.sparse.construct import hstack, vstack

from PyLOSt.algorithms.stitching.algorithm import Algorithm
from PyLOSt.algorithms.util.util_fit import fit2D, getXYGrid
from PyLOSt.algorithms.util.util_math import rmse
from PyLOSt.algorithms.util.util_reference import get_A_Ref
from PyLOSt.algorithms.util.util_stitching import binData, calc_inverse, correctAndJoin, flatten_list, getOverlaps2D, \
    get_A_allpix, get_sparse_from_dict, optimize_correctors


class GlobalOptimization(Algorithm):
    # use A x (C.T) = E
    # C = [[cxi][cyi]] correctors
    # E = [[exji][eyji]] pitch and roll errors

    name = 'global_optimize'
    description = 'Global optimization of overlap error function.'

    # Algo params
    showPlots = False
    data_type = ''
    method_optimization = 'least_squares_linear'  # least squares optimization
    min_overlap = 0.6  # ?? percent
    remove_outlier_subapertures = False
    initialization_advanced = False

    num_exclude_subaps = 0
    post_process = 'none'  # none/ref_cv

    # Corrector terms
    cor_piston = True
    cor_pitch = True
    cor_roll = True
    cor_reference_extract = False
    cor_all_pixels = False

    ref_extract_type = 'full'
    start_deg = 4
    end_deg = 10
    scale_xy = False
    init_to_average = False
    ref_diff_nstd = 0
    constraint_zero_sum = False
    bound_nstd = 0

    use_threshold = 'none'  # none,pre_process,post_process
    threshold_minval = 0
    threshold_maxval = 0

    use_binning_opt = False
    bin_size_x = 4
    bin_size_y = 4

    filt_bad_pix = False
    filt_bad_pix_threshold = 3  # filter >3 orders of std from ideal shape

    def __init__(self, stitch_options, data_in=None):
        """
        :param data_in: Data object containg raw data and other data needed for stitching
        :param stitch_options: Parameters for stitching, stitching algorithm
        """
        algo_options = stitch_options['algorithm_options']
        Algorithm.__init__(self, algo_options, data_in)
        self.data_out['creator'] = u'global_optimization.py'

    def pre_process(self, scan_item, otype):
        if self.use_threshold == 'pre_process':
            scan_item[scan_item > self.threshold_maxval] = np.nan
            scan_item[scan_item < self.threshold_minval] = np.nan

    def stitch_scan_item(self, key, scan_item, intensity, res_item, res_intensity, mX, mY, pix_size, prog_block=0):
        err_val = 0
        self.pix_size = pix_size
        C = np.array([[0] * 3] * len(mX))
        cor_terms = [self.cor_piston, self.cor_roll, self.cor_pitch]
        ref_scale = [1, 1]

        self.pre_process(scan_item, otype=key)
        sdiff, A, E, E_resd, validPat, slc_i, slc_j = getOverlaps2D(self, mX, mY, scan_item, key, pix_size,
                                                                    cor_terms=cor_terms,
                                                                    cor_ref=self.cor_reference_extract,
                                                                    showPlots=self.showPlots,
                                                                    prog_block=prog_block * 0.3)

        C, C_ref, ref_scale = self.getCorrectors(C, A, E, E_resd, sdiff, scan_item, key, mX, mY, pix_size, slc_i, slc_j,
                                                 res_item, prog_block=prog_block * 0.5)
        err_val, res_item, scan_item_cor, ref_ext = correctAndJoin(self, C, scan_item, pix_size, res_item,
                                                                   res_intensity, otype=key, mX=mX, mY=mY, C_ref=C_ref,
                                                                   ref_scale=ref_scale, prog_block=prog_block * 0.2)
        return err_val, res_item, C, scan_item_cor, ref_ext

    def getCorrectors(self, P0, A, E, E_resd, sdiff, scan_item, otype, mX, mY, pix_size, slc_i, slc_j, res_item,
                      prog_block=0):
        """
        Optimize for correctors where overlap error between subaperture is minimum

        :param P0: Initial correctors
        :param A: Matrix containing subapertures indices in a overlap - IndexMatrix
        :param sdiff: Overlap error array
        :param sarr_ret: Subaperture array
        :param otype: Data type slope(X/Y)/height
        :return: Optimized correctors
        """
        f_pix = 0.3
        f_ref = 0.3
        f = 1 - (f_pix if self.cor_all_pixels else 0) - (f_ref if self.cor_reference_extract else 0)
        nSA = len(mX)
        P_ref = None
        P = P0
        ref_scale = [1, 1]

        subap_avg = np.nanmean(scan_item, 0)
        self.subap_avg_rms = rmse(subap_avg)
        if self.initialization_advanced:
            A_coo = get_sparse_from_dict(A, shape=(len(A), len(mX)))
            A_coo1 = vstack([A_coo, [1] * A_coo.shape[1]])  # Append global pitch/roll/piston set to zero
            E1 = E + [[0, 0, 0]]
            P0 = calc_inverse(A_coo1, E1)
        else:
            P0[:, 0] = np.nanmean(np.nanmean(scan_item, axis=2), axis=1)  # TODO: why P0 is int32 ?
        P0_shape = P0.T.shape
        P0 = list(P0.T.flatten())
        if self.cor_all_pixels:
            if self.use_binning_opt:
                sdiff = {key: binData(sdiff[key], self.bin_size_x, self.bin_size_y) for key in sdiff}
            # Ef                          = sdiff.transpose(0,2,1).reshape(sdiff.shape[0],-1)
            Ef = np.asarray(flatten_list([list(sdiff[key][~np.isnan(sdiff[key])]) for key in sdiff]))[:, np.newaxis]
            A_coo = get_A_allpix(self, A, sdiff, mX, mY, self.pix_size, res_item, otype, prog_block=f_pix * prog_block)
        else:
            Ef = np.asarray(E).T.ravel()[:, np.newaxis]
            A_coo = get_sparse_from_dict(A, shape=(len(A), len(mX)), mode=2)

        if self.cor_reference_extract:
            E_resd = np.array(E_resd)[~np.isnan(E_resd)]
            B_coo, ref_scale = get_A_Ref(self, scan_item, sdiff, mX, mY, pix_size, slc_i, slc_j, otype,
                                         scaleXY=(self.scale_xy and not self.cor_all_pixels),
                                         prog_block=0.7 * f_ref * prog_block)
            P_ref = self.init_ref_correctors(scan_item, pix_size) if self.init_to_average else [0] * B_coo.shape[-1]
            if self.cor_all_pixels and A_coo.shape[0] == B_coo.shape[0]:
                A_coo = hstack([A_coo, B_coo])
                P0 += P_ref
            else:
                B_coo, E_resd = self.filt_ref_rows(B_coo, E_resd)
                P_ref = optimize_correctors(self, B_coo, E_resd, P_ref, self.method_optimization, show_cost=True)  # TODO: the major cost --> should be multithreaded

        # Append global pitch/roll/piston set to zero
        A_coo = vstack([A_coo, [1] * nSA + [0] * 2 * nSA + [0] * (A_coo.shape[-1] - 3 * nSA),
                        [0] * nSA + [1] * nSA + [0] * nSA + [0] * (A_coo.shape[-1] - 3 * nSA),
                        [0] * 2 * nSA + [1] * nSA + [0] * (A_coo.shape[-1] - 3 * nSA)])
        Ef = np.vstack([Ef, [0], [0], [0]])
        P = optimize_correctors(self, A_coo, Ef, P0, self.method_optimization,
                                show_cost=(self.cor_reference_extract and self.cor_all_pixels))
        P = np.array(P)
        if self.cor_reference_extract and self.cor_all_pixels:
            P_ref = P[3 * nSA:]
            P = P[:3 * nSA]
        P = P.reshape(P0_shape).T
        self.increment_progress(f * prog_block)
        return P, P_ref, ref_scale

    def filt_ref_rows(self, A, E):
        if self.ref_diff_nstd > 0:
            En = np.asarray(E)
            filt = np.abs(En - np.nanmean(En)) < self.ref_diff_nstd * rmse(En)
            E = En[filt]
            A = A.tocsr()[filt]
        return A, E

    def init_ref_correctors(self, scan_item, pix_sz):
        avg = np.nanmean(scan_item, 0)
        if self.ref_extract_type == 'full':
            _, resd, _ = fit2D(avg, pix_size=pix_sz, degree=2, start_degree=0, retResd=True)
            return list(resd.flatten())
        else:
            enDeg = self.end_deg
            stDeg = self.start_deg
            C0, _, _ = fit2D(avg, pix_size=pix_sz, degree=enDeg, start_degree=stDeg, retResd=False,
                             typ=self.ref_extract_type)
            return list(C0)

    @staticmethod
    def getXY(sarr, pix_sz, oXArr, oYArr):
        """
        Get XY grid of subapertures in mirror coordinates

        :param sarr:
        :param pix_sz:
        :param oXArr:
        :return:
        """
        res_mask_szY = sarr.shape[-2] + max(oYArr)
        res_mask_szX = sarr.shape[-1] + max(oXArr)
        mask_mir = np.full([res_mask_szY, res_mask_szX], True)
        xx, yy = getXYGrid(mask_mir, pix_size=pix_sz, order=2, mask=mask_mir)
        pcnt, ny, nx = sarr.shape
        xx_arr = np.zeros_like(sarr)
        yy_arr = np.zeros_like(sarr)
        for j, s in enumerate(sarr):
            ox = oXArr[j]
            oy = oYArr[j]
            slc = (slice(oy, oy + ny), slice(ox, ox + nx))
            xx_arr[j, :, :] = xx[slc]
            yy_arr[j, :, :] = yy[slc]
        return xx_arr, yy_arr

    def maskByThreshold(self, carr, oarr):
        """
        Apply a threshold on subaperture values

        :param carr: Corrected subapertures
        :param oarr: Original subapertures
        :return: Corrected subapertures after threshold
        """
        carr[oarr < self.threshold_minval] = np.nan
        carr[oarr > self.threshold_maxval] = np.nan
        return carr
