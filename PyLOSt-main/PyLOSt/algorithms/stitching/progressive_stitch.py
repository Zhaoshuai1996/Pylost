# coding=utf-8
'''
Created on Apr 9, 2018

Subapertures are progressively corrected for pitch/roll/piston errors with respect an intermediate stitching result and finally they are joined.

@author: ADAPA
'''
import copy

import numpy as np
from scipy import optimize

from PyLOSt.algorithms.stitching.algorithm import Algorithm
from PyLOSt.algorithms.util.util_fit import getXYGrid
from PyLOSt.algorithms.util.util_math import rms


class ProgressiveStitch(Algorithm):
    name = 'progressive_stitch'
    description = 'Successive joining of subapertures with pitch/roll corrected.'

    # Algorithm inputs
    data_type = {'value': '', 'description': 'Input data type', 'all_values': ('', 'slope', 'height'), 'disp_type': 'S',
                 'unit': None}
    use_least_squares = {'value': True, 'disp_type': 'C',
                         'description': 'Use least square optimization in the overlap. If selected uses all pixels in the overlap region in least square optimization. '
                                        'If not selected uses the mean difference for correction.'}

    def __init__(self, stitch_options, data_in=None):
        """
        :param data_in: Data object containg raw data and other data needed for stitching
        :param stitch_options: Parameters for stitching, stitching algorithm
        """
        algo_options = stitch_options['algorithm_options']
        Algorithm.__init__(self, algo_options, data_in)
        self.iterate_max_count = 5
        self.data_out['creator'] = u'progressive_stitch.py'

    def stitch_scan_item(self, key, scan_item, intensity, res_item, res_intensity, mX, mY, pix_size, prog_block=0):
        err_val = 0
        szItem = scan_item.shape
        totErr = [0.0] * self.iterate_max_count
        scan_item_cor = np.full(scan_item.shape, np.nan, dtype=scan_item.dtype)
        pres_item = np.full(res_item.shape, np.nan, dtype=res_item.dtype)
        pres_intensity = np.full(res_item.shape, np.nan, dtype=res_item.dtype)
        pscan_item_cor = np.full(scan_item.shape, np.nan, dtype=scan_item.dtype)
        pcorrectors = [[0]] * len(mX)
        new_block = prog_block * 1 / (self.iterate_max_count * len(mX))
        for i in range(0, self.iterate_max_count):
            res_item[:] = np.nan
            res_intensity[:] = np.nan
            correctors = [[0]] * len(mX)
            for j, ox in enumerate(mX):
                oy = mY[j]
                slc = (slice(oy, oy + szItem[-2]), slice(ox, ox + szItem[-1]))
                errVal, item_cj, cor_j = self.correct_j(key, scan_item, (res_item if i == 0 else pres_item),
                                                        (res_intensity if i == 0 else pres_intensity), j, slc, pix_size)
                res_item[slc] = np.nansum([res_item[slc], item_cj], axis=0)
                res_intensity[slc] = np.nansum([res_intensity[slc], intensity[j]], axis=0)
                totErr[i] = totErr[i] if np.isnan(errVal) else totErr[i] + errVal
                correctors[j] = cor_j
                scan_item_cor[j] = item_cj
                self.increment_progress(new_block)

            if i > 0 and totErr[i] > totErr[i - 1]:
                break
            print('Iteration {}'.format(i))
            np.copyto(pres_item, res_item)
            np.copyto(pres_intensity, res_intensity)
            np.copyto(pscan_item_cor, scan_item_cor)
            pcorrectors = copy.deepcopy(correctors)

        res_item = np.divide(pres_item, pres_intensity)
        err_val = self.get_algorithm_error(mX, mY, pscan_item_cor, res_item)
        return err_val, res_item, pcorrectors, scan_item_cor

    def correct_j(self, key, scan_item, res_item, res_intensity, j, slc, pix_size):
        item_j = scan_item[j]
        res_j = np.divide(res_item[slc], res_intensity[slc]) if np.any(res_intensity) else res_item[slc]
        res_j[np.isinf(res_j)] = np.nan

        errVal, item_cj, cor_j = self.corOverlpaErr(item_j, res_j, otype=key, pix_size=pix_size)

        return errVal, item_cj, cor_j

    def corOverlpaErr(self, item_j, res_j, otype, pix_size=1.0):
        """
        Correct j-th subaperture to intermediate (j-1)-th stitched result

        :param item_j: j-th subaperture
        :param res_j: (j-1)-th intermediate stitched result
        :param otype: Data type slope/height
        :return: Corrected subaperture
        """
        item_cj = item_j
        cor_j = [0]
        if otype in ['slopes_y', 'slopes_x']:
            # find c in M = cI+pat_o-res_o, such that
            if self.use_least_squares:
                # sum(M^2) is minimized -- least squares optimization
                p0 = 0  # initial guess
                index = ~(np.isnan(item_j) | np.isnan(res_j))
                if (np.any(item_j[index]) and np.any(res_j[index])):
                    cor_j, C, info, msg, success = optimize.leastsq(self.errfunc_slp, p0,
                                                                    args=(item_j[index], res_j[index]), full_output=1)
                    item_cj = cor_j + item_j
            else:
                cor_j = -np.nanmean((item_j - res_j).flatten())
                item_cj = cor_j + item_j
        elif otype == 'height':
            if self.use_least_squares:
                p0 = [0, 0, 0]
                index = ~(np.isnan(item_j) | np.isnan(res_j))
                if (np.any(item_j[index]) and np.any(res_j[index])):
                    xx, yy = getXYGrid(item_j, pix_size=pix_size, order=2, mask=np.ones_like(item_j))
                    cor_j, C, info, msg, success = optimize.leastsq(self.errfunc_h, p0, args=(
                        item_j[index], res_j[index], xx[index], yy[index]), full_output=1)
                    item_cj = cor_j[0] + cor_j[1] * yy + cor_j[2] * xx + item_j
        errVal = rms((item_cj - res_j).flatten())
        return errVal, item_cj, cor_j

    @staticmethod
    def errfunc_slp(p, t, y):
        return p + t - y

    @staticmethod
    def errfunc_h(p, t, y, xx, yy):  # p is array of coeff of c0+c1x+c2y
        return p[0] + p[1] * yy + p[2] * xx + t - y
