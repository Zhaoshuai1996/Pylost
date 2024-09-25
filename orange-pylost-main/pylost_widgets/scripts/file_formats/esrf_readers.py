# coding=utf-8

import numpy as np
from Orange.data import FileFormat

from PyLOSt.data_in.esrf.keyence import KeyenceData
from pylost_widgets.util.MetrologyData import MetrologyData


class KeyenceReader(FileFormat):
    """File reader for Zygo MetroPro/MX formats dat, datx"""
    EXTENSIONS = ('.vk4',)
    DESCRIPTION = 'Keyence file reader'
    SUPPORT_COMPRESSED = False
    SUPPORT_SPARSE_DATA = False
    PRIORITY = 3
    PARAMS = {'instr_scale_factor': 1.0}
    clear_output_before_loading = True

    def read(self):
        """
        Load Zygo dat or datx file.

        :return: Loaded data
        :rtype: dict
        """
        k = KeyenceData()
        hgt = k.readfile(self.filename)
        data = hgt.__dict__
        data['values'] = np.moveaxis(data['initial'].values, -1, -2)
        data['motorX'] = np.array(np.nan)
        data['motorY'] = np.array(np.nan)
        return data

    @staticmethod
    def data_standard_format(data):
        """
        Convert data in standard format readable by orange pylost widgets, e.g. import heights / slopes_x as MetrologyData from the raw file data.

        :param data: Raw file data
        :type data: dict
        :return: Standard format data
        :rtype: dict
        """
        ret_data = {}
        try:
            height = MetrologyData(data['values'],  # np.moveaxis(data['values'],-1,-2),
                                   unit='{}'.format(data['units']['values']),
                                   pix_size=data['header']['lateral_res'],
                                   pix_unit='{}'.format(data['units']['pixel']), dim_detector=[-2, -1],
                                   axis_names=['Motor', 'Y', 'X']).to('nm')
            ret_data = {'height': height}
        except Exception as e:
            raise Exception('Error while converting to standard format: {}'.format(repr(e)))
        finally:
            return ret_data
