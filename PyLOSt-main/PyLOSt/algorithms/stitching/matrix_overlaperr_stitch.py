# coding=utf-8
'''
Created on Mar 9, 2019

Overlaps are fit to plane and the piston/pithc/roll errors between the subapertures are extracted.
From this data piston/pithc/roll correction needed for each subaperture is determined using matrix techniques of solving linear equations.

@author: ADAPA
'''

import numpy as np
from scipy import signal
from scipy.sparse.construct import hstack, vstack

from PyLOSt.algorithms.stitching.algorithm import Algorithm
from PyLOSt.algorithms.util.util_reference import get_A_Ref
from PyLOSt.algorithms.util.util_stitching import calc_inverse, correctAndJoin, flatten_list, getOverlaps2D, \
    get_A_allpix, get_sparse_from_dict, plotSt


class MatrixOverlapErrStitch(Algorithm):
    # use A x (C.T) = E
    # C = [[cxi][cyi]] correctors
    # E = [[exji][eyji]] pitch and roll errors

    name = 'matrix_overlaperr'
    description = 'Overlap errors between all subapertures are solved through matrix based linear equations solutions.'

    # show intermediate plots
    showPlots = False
    DEG_TO_MRAD = 17.4533
    num_exclude_subaps = 0

    ### Algo params
    data_type = ''
    inv_type = 'SVD'
    min_overlap = 0.6  # 60 percent
    remove_outlier_subapertures = False
    post_process = 'none'  # none/ref_cv

    # Corrector terms
    cor_piston = True
    cor_pitch = True
    cor_roll = True
    cor_reference_extract = False
    cor_all_pixels = False

    ref_extract_type = 'poly'
    start_deg = 4
    end_deg = 10
    scale_xy = False

    use_threshold = 'none'  # none,pre_process,post_process
    threshold_minval = 0
    threshold_maxval = 0

    filt_bad_pix = False
    filt_bad_pix_threshold = 3  # filter >3 orders of std from ideal shape

    def __init__(self, stitch_options, data_in=None):
        """
        :param data_in: Data object containg raw data and other data needed for stitching
        :param stitch_options: Parameters for stitching, stitching algorithm
        """
        algo_options = stitch_options['algorithm_options']
        Algorithm.__init__(self, algo_options, data_in)
        self.data_out['creator'] = u'matrix_overlaperr_stitch.py'

    def stitch_scan_item(self, key, scan_item, intensity, res_item, res_intensity, mX, mY, pix_size, prog_block=0):
        err_val = 0
        ref_scale = [1, 1]
        correctors = [[0]] * len(mX)
        cor_terms = [self.cor_piston, self.cor_roll, self.cor_pitch]

        if ~np.any(~np.isnan(scan_item)):
            return res_item, correctors

        self.pre_process(scan_item, otype=key)
        sdiff, A, E, E_resd, validPat, slc_i, slc_j = getOverlaps2D(self, mX, mY, scan_item, key, pix_size,
                                                                    cor_terms=cor_terms,
                                                                    cor_ref=self.cor_reference_extract,
                                                                    showPlots=self.showPlots,
                                                                    prog_block=prog_block * 0.3)
        if ~np.any(validPat):
            return res_item, correctors

        C, C_ref, ref_scale = self.calculate_corrections(A, E, E_resd, validPat, scan_item, sdiff, mX, mY, pix_size,
                                                         res_item, slc_i, slc_j, otype=key, prog_block=prog_block * 0.5)
        err_val, res_item, scan_item_cor, ref_ext = correctAndJoin(self, C, scan_item, pix_size, res_item,
                                                                   res_intensity, otype=key, mX=mX, mY=mY, C_ref=C_ref,
                                                                   ref_scale=ref_scale, prog_block=prog_block * 0.2)
        if self.verbose > 0:
            print('Fin correction')

        return err_val, res_item, C, scan_item_cor, ref_ext

    def pre_process(self, scan_item, otype):
        if self.use_threshold == 'pre_process':
            scan_item[scan_item > self.threshold_maxval] = np.nan
            scan_item[scan_item < self.threshold_minval] = np.nan

    def calculate_corrections(self, A, E, E_resd, validPat, scan_item, sdiff, mX, mY, pix_size, res_item, slc_i, slc_j,
                              otype, prog_block=0):
        ref_scale = [1, 1]
        A_coo = get_sparse_from_dict(A, shape=(len(A), len(mX)))
        # Append global pitch/roll/piston set to zero
        A_coo = vstack([A_coo, [1] * A_coo.shape[1]])
        E += [[0, 0, 0]]

        f_pix = 0.3
        f_ref = 0.5
        f = 1 - (f_pix if self.cor_all_pixels else 0) - (f_ref if self.cor_reference_extract else 0)
        nPat = len(mX)
        C_ref = None
        if self.cor_all_pixels:
            A_coo = get_A_allpix(self, A, sdiff, mX, mY, pix_size, res_item, otype, prog_block=f_pix * prog_block)
            E = np.asarray(flatten_list([list(sdiff[key][~np.isnan(sdiff[key])]) for key in sdiff]))
        if self.cor_reference_extract:
            E_resd = np.array(E_resd)[~np.isnan(E_resd)]
            B_coo, ref_scale = get_A_Ref(self, scan_item, sdiff, mX, mY, pix_size, slc_i, slc_j, otype,
                                         scaleXY=(self.scale_xy and not self.cor_all_pixels),
                                         prog_block=0.7 * f_ref * prog_block)
            if self.cor_all_pixels and A_coo.shape[0] == B_coo.shape[0]:
                A_coo = hstack([A_coo, B_coo])
            else:
                C_ref = calc_inverse(self, B_coo, E_resd)
            self.increment_progress(0.3 * f_ref * prog_block)

        C = calc_inverse(self, A_coo, E)
        if self.cor_all_pixels:
            if self.cor_reference_extract:
                C_ref = C[3 * nPat:]
                C = C[:3 * nPat]
                C = np.reshape(C, (3, nPat)).T
            else:
                C = np.reshape(C, (3, nPat)).T
        if self.verbose > 0:
            print('Fin determining correctors array')

        if self.post_process == 'ref_cv':
            C[validPat, 1:3] = signal.detrend(C[validPat, 1:3], axis=0)
        if self.showPlots:
            plotSt(C[:, 0], pnum=6, title='Piston')
            plotSt(C[:, 1], pnum=7, title='Roll')
            plotSt(C[:, 2], pnum=8, title='Pitch')
        self.increment_progress(f * prog_block)

        return C, C_ref, ref_scale

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
