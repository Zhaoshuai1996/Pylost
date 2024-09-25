# coding=utf-8
from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.util_functions import MODULE_SINGLE, MODULE_MULTI
from pylost_widgets.widgets._PylostBase import PylostBase
import numpy as np


class Operator(PylostBase):
    """
    Base class for operator widgets like addition, subtraction, multiplication, division
    """

    NONE, CENTER, LEFT, RIGHT, OFFSET_START = range(5)
    ALIGN_OPT = ['None', 'Center', 'Left', 'Right', 'Offset start position']

    def __init__(self):
        """Initialize class"""
        super().__init__()
        self.data_index = []

    # def set_data(self, data, id):
    #     self.clear_messages()
    #     if id in self.data_in:
    #         if data is None:
    #             del self.data_in[id]
    #         else:
    #             self.data_in[id] = data
    #     else:
    #         if data is not None:
    #             self.data_in[id] = data
    #
    #     self.init_data()

    def init_data(self):
        """Initialize data channel and process any new data in the input data channel"""
        self.data_in = {}
        for idx, data in enumerate(self.data_index):
            if data is not None:
                self.data_in[idx] = data

        self.update_data_names(self.data_in, multiple=True)
        if len(self.data_in) > 0:
            self.update_input_modules(self.data_in, multiple=True)
            self.load_data()
        else:
            self.data_out = {}
            self.Outputs.data.send(None)
            self.infoInput.setText("No data on input yet, waiting to get something.")

    def update_comment(self, comment='', prefix=''):
        """
        Update comment text, shown as info_message and also forwarded to next widget as log text

        :param comment: Comment text
        :type comment: str
        :param prefix: Prefix text for comment
        :type prefix: str
        """
        # cmt = ''
        # for key in self.data_in:
        #     if 'comment_log' in self.data_in[key]:
        #         cmt += '{} link {} :\n{}\n\n'.format(self.name, key+1, self.data_in[key]['comment_log'].replace('\n', '\n\t'))
        # self.data_out['comment_log'] = cmt + "\n{} operation : {}".format(self.name, comment)
        super().update_comment(comment, '{} operation'.format(self.name))

    def load_data(self, multi=False):
        """Load data method called after new data arrives in the widget. Implementation in super class PylostBase is called, subsequently load_module is called."""
        super().load_data(multi=True)
        self.load_module()

    def load_module(self):
        """Central function which differentiates the modules of different data inputs. The operator is typically applied for multiple inputs together (e.g. add tow inputs).
        If the different inputs do not belong to same module, appropriate logic needs to be applied for applying operator.

        E.g. input 1 is scan_data module, i.e. it has n scans each containing a height or slopes data object,
        input 2 is custom module with single height/slope data object. The second input data object is added/subtracted... from each of the scan in first input"""
        try:
            self.clear_messages()
            self.data_out = {}
            result = {}
            self.module_data = self.get_data_by_module(self.data_in, None, multiple=True)
            for key in self.module_data:
                id, module = key
                data_id = self.module_data[key]
                if not isinstance(data_id, dict):
                    continue  # Only operate on data loaded in dict. Additional info such as comment strings are ignored
                if len(result) == 0:
                    # copy_items(data_id, result)
                    if module in MODULE_MULTI:
                        result = {x: self.apply_scan({}, data_id[x]) for x in data_id}
                    elif module in MODULE_SINGLE:
                        result = self.apply_scan({}, data_id)
                    result['module'] = module
                else:
                    result_temp = {}
                    if result['module'] in MODULE_SINGLE:  # any(set(self.DATA_NAMES).intersection(result.keys())):
                        # Current result is single
                        if module in MODULE_MULTI:
                            # New module is many scans
                            result_temp = {x: self.apply_scan(result, data_id[x]) for x in data_id}
                        elif module in MODULE_SINGLE:
                            # New module is single
                            result_temp = self.apply_scan(result, data_id)
                        result_temp['module'] = module
                    else:
                        # Current result is many scans
                        if module in MODULE_MULTI:
                            # New module is many scans
                            result_temp = {x: self.apply_scan(result[x], data_id[x]) for x in result if x in data_id}
                        elif module in MODULE_SINGLE:
                            # New module is single
                            result_temp = {x: self.apply_scan(result[x], data_id) for x in result}
                    result = result_temp
            if 'module' in result:
                del result['module']
            self.set_data_by_module(self.data_out, self.module, result)
            self.update_comment('for {} input elements'.format(len(self.module_data)))
            self.Outputs.data.send(self.data_out)
        except Exception as e:
            self.Outputs.data.send(None)
            self.Error.unknown(repr(e))

    def apply_scan(self, scan_result, scan=None, comment=''):
        """
        Re implement the apply_scan (defined in super class PylostBase) for operators. Actual implementation will be in the subclasses of this class.

        :param scan_result: The result scan. It is updated each time apply_scan is called for multiple inputs and possibly many scans
        :type scan_result: dict
        :param scan: Current scan data to apply
        :type scan: dict
        :param comment: Comment text
        :type comment: str
        """
        pass

    def pad_items(self, Z1, Z2, align_type=NONE):
        if align_type == self.NONE:
            return Z1, Z2
        else:
            if Z1.ndim != Z2.ndim:
                if isinstance(Z1, MetrologyData) and isinstance(Z2, MetrologyData):
                    dims1 = super().get_detector_dimensions(Z1)
                    axes1 = dims1.nonzero()[0]
                    dims2 = super().get_detector_dimensions(Z2)
                    axes2 = dims2.nonzero()[0]
                    if len(axes1) > 0 and len(axes1) == len(axes2):
                        p1 = [(0, 0)] * Z1.ndim
                        p2 = [(0, 0)] * Z2.ndim
                        for ax1, ax2 in zip(axes1, axes2):
                            st1 = Z1.index_list[ax1][0] if align_type == self.OFFSET_START else 0
                            st2 = Z2.index_list[ax2][0] if align_type == self.OFFSET_START else 0
                            p1[ax1] = self.get_pads_z1(Z1.shape[ax1], Z2.shape[ax2], st1, st2, align_type)
                            p2[ax2] = self.get_pads_z2(Z1.shape[ax1], Z2.shape[ax2], st1, st2, align_type)
                        Z1 = np.pad(Z1, p1, mode='constant', constant_values=np.nan)
                        Z2 = np.pad(Z2, p2, mode='constant', constant_values=np.nan)
                    else:
                        raise Exception('Shape mismatch between items. Unable to pad nans')
                else:
                    raise Exception('Shape mismatch between items. Unable to pad nans')
            else:
                st1a = [Z1.index_list[i][0] if align_type == self.OFFSET_START else 0 for i in range(Z1.ndim)]
                st2a = [Z2.index_list[i][0] if align_type == self.OFFSET_START else 0 for i in range(Z1.ndim)]
                p1 = [self.get_pads_z1(Z1.shape[i], Z2.shape[i], st1a[i], st2a[i], align_type) for i in range(Z1.ndim)]
                p2 = [self.get_pads_z2(Z1.shape[i], Z2.shape[i], st1a[i], st2a[i], align_type) for i in range(Z1.ndim)]
                Z1 = np.pad(Z1, p1, mode='constant', constant_values=np.nan)
                Z2 = np.pad(Z2, p2, mode='constant', constant_values=np.nan)
            return Z1, Z2

    def get_pads_z1(self, sz1, sz2, st1, st2, align_type):
        m = min(st1, st2)
        st1 = st1 - m
        st2 = st2 - m
        if st1 + sz1 < st2 + sz2:
            if align_type == self.OFFSET_START:
                align_type = self.LEFT
                p = self.get_pads(st2 + sz2 - st1 - sz1, align_type)
                return (st1, p[1])
            else:
                return self.get_pads(st2 + sz2 - st1 - sz1, align_type)
        else:
            return (st1, 0)

    def get_pads_z2(self, sz1, sz2, st1, st2, align_type):
        m = min(st1, st2)
        st1 = st1 - m
        st2 = st2 - m
        if st2 + sz2 < st1 + sz1:
            if align_type == self.OFFSET_START:
                align_type = self.LEFT
                p = self.get_pads(st1 + sz1 - st2 - sz2, align_type)
                return (st2, p[1])
            else:
                return self.get_pads(st1 + sz1 - st2 - sz2, align_type)
        else:
            return (st2, 0)

    def get_pads(self, diff, align_type):
        p = (0, 0)
        if align_type == self.LEFT:
            p = (0, diff)
        elif align_type == self.RIGHT:
            p = (diff, 0)
        elif align_type == self.CENTER:
            left = int((diff) / 2)
            right = diff - left
            p = (left, right)
        return p
