# coding=utf-8
import datetime

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from astropy.units import Quantity

from PyLOSt.algorithms.util.util_algo import differentiate_heights, get_default_data_names
from PyLOSt.algorithms.util.util_fit import fit2D, getPixSz2D, getXYGrid
from PyLOSt.algorithms.util.util_math import rms


class Algorithm(QObject):
    name = ''
    description = ''
    ctype = 'S'  # optional, default is 'S', choices: ['S':'Stitching', 'NS':'Non stitchng']
    added_by = None  # optional, e.g. user name
    location = None  # optional, e.g. 'ESRF'

    ## Options are added here ##
    ## Options with dropdowns are added as tuple ##
    ## Options with units can be added using 'astropy: Quantity' (https://docs.astropy.org/en/stable/install.html) ##
    ## Options with loading external file can be set to _file_option ##
    _file_option = 'file:'
    finished = pyqtSignal()
    output = pyqtSignal(dict)
    progress = pyqtSignal(float)
    info = pyqtSignal(str)
    cost = pyqtSignal(float)

    def __init__(self, options, data_in=None, **kwargs):
        QObject.__init__(self)
        attributes = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith('_')]
        for attr in attributes:
            attr_default = getattr(self, attr, None)
            if type(attr_default) is dict and 'value' in attr_default:
                attr_val = attr_default['value']
            else:
                attr_val = attr_default

            if attr in options:
                setattr(self, attr, options[attr])
            elif type(attr_val) is tuple:
                setattr(self, attr, attr_val[0])

        self.DEFAULT_DATA_NAMES = get_default_data_names()
        self.data_in = data_in
        self.data_out = {}
        self.progressBarSet = None
        self.cur_prog = 0
        self.callbackCancel = None

        self.verbose = 1
        self.use_numba = False

    def increment_progress(self, val):
        """
        Update progressbar value and emit it to be captured by main thread

        :param val: increment value
        :type val: float
        """
        self.cur_prog = self.cur_prog + val
        self.progress.emit(self.cur_prog * 100)
        self.check_cancel()

    def set_progress(self, val):
        """
        Update progressbar value and emit it to be captured by main thread

        :param val: new value
        :type val: float
        """
        self.cur_prog = val
        self.progress.emit(self.cur_prog * 100)
        self.check_cancel()

    def set_info(self, val, typ=''):
        """Set info and costfucntion with new value"""
        try:
            if typ == 'costfunc':
                self.cost.emit(val)
            else:
                self.info.emit(val)
        except Exception as e:
            print(e)

    def stitch(self, callback=None, **kwargs):
        """Stitch input data"""
        # profiler = start_profiler()
        import time
        start = time.time()
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        if 'use_numba' in kwargs:
            self.use_numba = kwargs['use_numba']
        self.callbackCancel = callback
        data = self.stitchImages()
        self.output.emit(data)
        end = time.time()
        print('Total stitch time: {} seconds'.format(end - start))
        # print_profiler(profiler)
        return data

    def finish(self):
        """Emit stitching finish signal"""
        self.finished.emit()

    def check_cancel(self):
        """Check if cancel/abort button is pressed"""
        if self.callbackCancel is not None:
            self.callbackCancel()

    def stitchImages(self):
        """
        Stitching function for 2d images. Stitching either in 1D or 2D
        """
        self.data_out['stitched_scans'] = {}
        self.data_out['stitched_on'] = str(datetime.datetime.now())

        instr_scale = self.data_in['instr_scale_factor'] if 'instr_scale_factor' in self.data_in else 1.0
        pix_size = self.data_in['pix_size'] if 'pix_size' in self.data_in else 1.0

        if 'scan_data' in self.data_in:
            scans = self.data_in['scan_data']
            for i, it in enumerate(scans):
                scan = scans[it]
                self.data_out['stitched_scans'][it] = self.stitch_scan(scan, instr_scale, pix_size,
                                                                       prog_block=1 / len(scans))
        elif any(set(self.DEFAULT_DATA_NAMES).intersection(self.data_in.keys())):
            self.data_out['stitched_scans'] = self.stitch_scan(self.data_in, instr_scale, pix_size)  # ['scan_custom']
        return self.data_out

    def stitch_scan(self, scan, instr_scale=1.0, pix_size=1.0, prog_block=1):
        """
        Stitch each scan separately.

        :param scan: Scan to be stitched
        :type scan: dict
        :param instr_scale: Instrument scaling factor
        :type instr_scale: float
        :param pix_size: Pixel size
        :type pix_size: float / array of floats / quantity
        :return: Stitched scan results as a dictionary
        """
        stitched_scan = {}
        if self.data_type == '':
            NAMES = self.DEFAULT_DATA_NAMES
        elif self.data_type == 'slope':
            NAMES = ['slopes_y', 'slopes_x']
        elif self.data_type == 'height':
            NAMES = ['height']
        else:
            NAMES = []
        keys = list(set(NAMES).intersection(scan.keys()))
        mX = scan['motor_X_pix']  # pixel X offsets from start
        mY = scan['motor_Y_pix'] if 'motor_Y_pix' in scan else np.full_like(mX, 0, dtype=int)
        pix_size = scan['pix_size'] if 'pix_size' in scan else pix_size
        pix_unit = ['', '']

        new_block = prog_block * 1 / len(keys) if len(keys) > 0 else prog_block
        for i, key in enumerate(keys):
            scan_item = scan[key]
            pix_size = getattr(scan_item, 'pix_size_detector', getPixSz2D(pix_size))
            pix_unit = [str(x.unit) if isinstance(x, Quantity) else x for x in pix_size]
            pix_size = [x.value if isinstance(x, Quantity) else x for x in pix_size]
            szItem = scan_item.shape
            res_szX = szItem[-1] + np.nanmax(mX)  # result size X stitch dirctn
            res_szY = szItem[-2] + np.nanmax(mY)  # result size Y
            intensity = scan['intensity'] / np.nanmax(
                scan['intensity'].flatten()) if 'intensity' in scan else np.invert(np.isnan(scan_item)).astype(int)
            res_item = np.full([res_szY, res_szX], np.nan)
            res_intensity = np.full([res_szY, res_szX], np.nan)

            scan_item_val = scan_item.value if scan_item.__class__.__name__ == 'MetrologyData' else scan_item
            out = self.stitch_scan_item(key, scan_item_val, intensity, res_item, res_intensity, mX, mY, pix_size,
                                        prog_block=new_block * 0.9)
            if len(out) > 0:
                stitched_scan[key + '_err_val'] = out[0]
            if len(out) > 1:
                result = instr_scale * out[1]

                try:
                    mask_mir = np.full_like(res_item, True)
                    xx_mir, yy_mir = getXYGrid(mask_mir, pix_size=pix_size, order=2, mask=mask_mir)
                    # Remove polynomial of deg 1
                    if hasattr(self, 'remove_tilts_from_stitched_image') and self.remove_tilts_from_stitched_image:
                        deg = 1 if key == 'height' else 0
                        _, result, _ = fit2D(result, pix_size=pix_size, degree=deg, retResd=True, xv=xx_mir, yv=yy_mir)

                    # Calculate radius of curvature
                    # terms                               = [1]+ [0]*5 + [1]   # [piston, ty, tx, cy, ast, cx, sph]
                    # cf, _, _                            = fit2D(result, pix_size=pix_size, filter_terms_poly=terms, retResd=False, xv=xx_mir, yv=yy_mir)
                    # rc                                  = (0.5 if key=='height' else 1) / cf[6]
                    # if scan_item.__class__.__name__ == 'MetrologyData' and pix_unit[-1]!='':
                    #     if key=='height':
                    #         rc = Quantity(rc, unit=units.Unit(pix_unit[-1])**2 / scan_item.unit).to('m')
                    #     elif key in ['slopes_x', 'slopes_y']:
                    #         rc = Quantity(rc, unit=units.Unit(pix_unit[-1]) / scan_item.unit).to('m', equivalencies=units.dimensionless_angles())
                    # stitched_scan[key + '_radius']      = rc
                except Exception as e:
                    print(e)

                if scan_item.__class__.__name__ == 'MetrologyData':
                    result = scan_item.copy_to_detector_format(result)
                stitched_scan[key] = result
            if len(out) > 2 and out[2] is not None:
                # stitched_scan[key+'_correctors']    = np.transpose(np.asarray(out[2]))
                correctors = np.asarray(out[2])
                if correctors.shape[1] > 0:
                    stitched_scan[key + '_piston'] = correctors[:, 0]
                if correctors.shape[1] > 1:
                    stitched_scan[key + '_roll'] = correctors[:, 1]
                if correctors.shape[1] > 2:
                    stitched_scan[key + '_pitch'] = correctors[:, 2]
            if len(out) > 3 and out[3] is not None:
                result = out[3]
                if scan_item.__class__.__name__ == 'MetrologyData':
                    result = scan_item.copy_to(result)
                stitched_scan[key + '_corrected'] = result
            if len(out) > 4 and out[4] is not None:
                ref_ext = -1 * out[4]  # minus 1 since this correction is added to subapertures to get final result
                if scan_item.__class__.__name__ == 'MetrologyData':
                    ref_ext = scan_item.copy_to_detector_format(ref_ext)
                    ref_ext._set_index_list(scan_item.index_list_detector)
                    # stitched_scan[key+'_features']              = self.get_features(scan_item, ref_ext, key)
                stitched_scan[key + '_reference_extracted'] = ref_ext

            self.increment_progress(new_block * 0.1)
        return stitched_scan

    def stitch_scan_item(self, key, scan_item, intensity, res_item, res_intensity, mX, mY, pix_size, prog_block=0):
        """
        Stitch a scan item, e.g. height. Implemented in subclasses

        :param key: item name, e.g. height
        :param scan_item: item value
        :param intensity: intensity value
        :param res_item: result stitched item initialized with nans
        :param res_intensity: result intensity
        :param mX: motor x positions
        :param mY: motor y positions
        :param pix_size: pixel size
        :return: implemnted in subclasses
        """
        pass

    @staticmethod
    def get_algorithm_error(mX, mY, scan_item_cor, res_item):
        """Get residual error after stitching"""
        szItem = scan_item_cor.shape
        errArr = []
        for j, ox in enumerate(mX):
            oy = mY[j]
            slc = (slice(oy, oy + szItem[-2]), slice(ox, ox + szItem[-1]))
            errArr += list(res_item[slc].ravel() - scan_item_cor[j].ravel())
        return rms(np.asarray(errArr))

    @staticmethod
    def get_features(scan_item, ref, key):
        """GEt feature of stitching to be used for data mining. Not used"""
        features = {}
        xx, yy = getXYGrid(scan_item, pix_size=[1, 1], order=3)
        features['x'] = xx + scan_item.start_position_pix[-1]
        features['y'] = yy + scan_item.start_position_pix[-2]
        val = np.nanmean(scan_item, axis=0)
        features[key] = val
        if key == 'height':
            features['slopes_x'], features['slopes_y'] = differentiate_heights(val, pix_sz=scan_item.pix_size_detector,
                                                                               method='grad')
        features['instr_err'] = ref
