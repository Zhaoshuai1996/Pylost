# coding=utf-8
import os
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
from Orange.data.io import class_from_qualified_name
from Orange.data import Table
from pylost_widgets.util.util_functions import stack_dict


class FileSeqLoader(QObject):
    """
    Class to load sequence of raw instrument files
    """
    progress = pyqtSignal(float)

    def load_scans(self, callback=None, scan_files=np.array([]), reader=None, block_size=100):
        """
        Load sequence of subapertures for a sequence of scans.

        :param callback: Callback function used to close progressbar
        :type callback: typing.Callable
        :param scan_files: A 2d array of file names with shape nb_scans x nb_subapertures
        :type scan_files: np.ndarray[list[str]]
        :param reader: File reader for the specified files
        :type reader: Orange.data.FileFormat
        :param block_size: Chunk size between 0 - 100, used to update work progress
        :type block_size: float
        :return: Loaded scan data
        :rtype: dict
        """
        res_data = {}
        for row in np.arange(scan_files.shape[0]):
            cur_prog = block_size * row / scan_files.shape[0]
            res_data['Scan_{}'.format(row)] = self.load_file_seq(filename_list=scan_files[row],
                                                                 reader=reader,
                                                                 block_size=block_size / scan_files.shape[0],
                                                                 cur_prog=cur_prog)
        return res_data

    def load_file_seq(self, callback=None, filename_list=[], reader=None, block_size=100, cur_prog=0):
        """
        Load sequence of subapertures.

        :param callback: Callback function used to close progressbar
        :type callback: typing.Callable
        :param filename_list: List of file names in the sequence
        :type filename_list: list[str]
        :param reader: File reader for the specified files
        :type reader: Orange.data.FileFormat
        :param block_size: Chunk size between 0 - 100, used to update work progress
        :type block_size: float
        :param cur_prog: Current progress
        :type cur_prog: float
        :return: Loaded sequence data
        :rtype: dict
        """
        res_data = {}
        start_pos_keys = reader.get_start_pos_keys() if getattr(reader, 'get_start_pos_keys', None) else []
        stack_selected_keys = reader.get_stack_selected_keys() if getattr(reader, 'get_stack_selected_keys',
                                                                          None) else []
        merge_selected_keys = reader.get_merge_selected_keys() if getattr(reader, 'get_merge_selected_keys',
                                                                          None) else []
        cam_size_keys = reader.get_cam_size() if getattr(reader, 'get_cam_size', None) else []
        for i, filename in enumerate(filename_list):
            self.progress.emit(cur_prog + i * block_size / len(filename_list))
            if not os.path.isfile(filename):
                continue
            if os.path.splitext(filename)[1].upper() not in [x.upper() for x in reader.EXTENSIONS]:
                continue
            qname = reader.qualified_name()
            reader_class = class_from_qualified_name(qname)
            reader = reader_class(filename)
            data = reader.read()
            if type(data) is dict:
                data_in = data
            else:
                data_in = {'file_data': np.array(data) if type(data) is Table else data}
            start_pos_keys, start_pos = stack_dict(data_in, res_data, start_pos_keys=start_pos_keys,
                                                   stack_selected_keys=stack_selected_keys,
                                                   cam_size_keys=cam_size_keys,
                                                   merge_selected_keys=merge_selected_keys)
            if start_pos is not None:
                res_data['start_pos'] = start_pos
        return res_data
