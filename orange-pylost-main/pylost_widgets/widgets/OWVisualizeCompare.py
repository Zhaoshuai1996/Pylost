# coding=utf-8

from Orange.widgets import gui
from Orange.widgets.utils.signals import Output
from Orange.widgets.widget import OWWidget
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QSizePolicy as Policy, QTabWidget
from orangewidget.settings import Setting
from orangewidget.utils.signals import MultiInput
from orangewidget.widget import Msg

from pylost_widgets.util.util_functions import MODULE_MULTI, MODULE_SINGLE, copy_items
from pylost_widgets.widgets.OWVisCompareBase import VisCompareBase
from pylost_widgets.widgets._PylostBase import PylostWidgets


class OWVisualizeCompare(PylostWidgets, VisCompareBase):
    name = 'Comparison 2D/1D'
    description = 'Compare datasets in 2D/1D visualizations.'
    icon = "../icons/vis.svg"
    priority = 34

    class Inputs:
        data = MultiInput('data', dict, auto_summary=False)

    class Outputs:
        lines = Output('selected lines', dict, default=True, auto_summary=False)
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    scan_name = Setting('')
    show_2d = Setting(True, schema_only=True)
    show_1d = Setting(True, schema_only=True)
    x_origin = Setting(False, schema_only=True)
    normalize = Setting(False, schema_only=True)
    level_data = Setting(True, schema_only=True)

    legend_names = Setting([], schema_only=True)
    count_links = Setting(0, schema_only=True)

    show_ellipse = Setting(True, schema_only=True)
    show_slopes_x = Setting(True, schema_only=True)
    show_slopes_y = Setting(True, schema_only=True)
    show_height = Setting(True, schema_only=True)
    show_rms = Setting(True, schema_only=True)
    show_pv = Setting(True, schema_only=True)
    show_rc = Setting(True, schema_only=True)
    show_line_stats = Setting(False, schema_only=True)

    manipulations = Setting([], schema_only=True)
    line_width = Setting(1, schema_only=True)
    line_pos = Setting([], schema_only=True)
    sel_lines = Setting({}, schema_only=True)

    class Error(OWWidget.Error):
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        self.has_updates = False
        # self.show()
        VisCompareBase.__init__(self)

        self.interpolate = True
        self.current_idx = 0

        box = super().init_info(module=True, module_callback=self.change_module, scans=True,
                                scans_callback=self.change_scan)
        self.btnReload = gui.button(box, self, 'Reload', callback=self.update_data,
                                    sizePolicy=(Policy.Fixed, Policy.Fixed))
        self.btnReload.hide()
        self.selModule.parent().hide()

        box = gui.vBox(self.controlArea, "Data viewer", stretch=19)
        self.main_tabs = QTabWidget(self)
        box.layout().addWidget(self.main_tabs)
        self.main_tabs.tabBarClicked.connect(self.click_main_tabs)

        self.reload_count = self.count_links

        # self._executor = ThreadExecutor()
        # task = Task()
        # def callback():
        #     if task.cancelled:
        #         pass
        #
        # self._executor.submit(partial(self.init_compare_stack, callback))

    def sizeHint(self):
        return QSize(1000, 550)

    @Inputs.data
    def set_data(self, index, data):
        self.data_index[index] = data
        super().init_data()

    @Inputs.data.insert
    def insert_data(self, index, data):
        self.data_index.insert(index, data)
        if self.reload_count <= 0:
            self.count_links += 1
            self.load_title(data, index)
        else:
            self.reload_count -= 1
        super().init_data()
        # print('insert_data: index={}, count_links={}'.format(index, self.count_links))

    @Inputs.data.remove
    def remove_data(self, index):
        self.count_links -= 1
        self.data_index.pop(index)
        self.legend_names.pop(index)
        super().init_data()
        # print('remove_data: index={}, count_links={}'.format(index, self.count_links))

    def activateWindow(self):
        self.update_data()
        super().activateWindow()

    def show(self):
        self.update_data()
        super().show()

    def focusInEvent(self, event):
        self.update_data()
        super().focusInEvent(event)

    def update_data(self):
        # profiler = start_profiler()
        self.clear_messages()
        if self.has_updates:
            self.data_out = {}
            self.update_input_modules(self.data_in, multiple=True)
            copy_items(self.data_in, self.data_out, deepcopy=True)
            self.update_tabs(show_all_default_names=True)
            self.Outputs.data.send(self.data_out if any(self.data_out) else None)
            if len(self.data_in) > 0:
                self.load_data(compare=True, all=False)
                vals = [self.id_name_map[x] for x in self.id_name_map if x in self.data_in]
                self.setStatusMessage('Showing plots - legends: {}'.format(vals))
            else:
                self.infoInput.setText("No data on input yet, waiting to get something.")
                self.no_data()
                self.Outputs.lines.send(None)
            self.has_updates = False
            self.btnReload.hide()
        # print_profiler(profiler)

    def load_data(self, **kwargs):
        try:
            super().load_data(multi=True)
            self.change_module(**kwargs)
            if len(self.manipulations) > 0:
                for manipulation in self.manipulations:
                    self.common_legend_events(manipulation)
        except Exception as e:
            self.Error.unknown(repr(e))

    def change_module(self, **kwargs):
        self.selScan.setEnabled(False)
        self.selScan.clear()
        module_data = self.get_data_by_module(self.data_out, None, multiple=True)
        keys = []
        for id in module_data:
            if id[1] in MODULE_MULTI:
                keys += list(set(module_data[id].keys()) - set(keys))
        if self.module in MODULE_MULTI:
            if len(module_data) > 0:
                self.selScan.setEnabled(True)
                self.selScan.addItems(keys)
                self.change_scan(**kwargs)
        elif self.module in MODULE_SINGLE:
            scan_vis = self.full_dataset(module_data, multiple=True)
            super().set_data_by_module(self.data_out, self.module, scan_vis, multiple=True)
            self.load_tabs_data(scan_vis, **kwargs)

    def no_data(self):
        self.selModule.clear()
        self.selScan.clear()
        self.data_out = {}

    def change_scan(self, **kwargs):
        curScan = self.selScan.currentText()
        module_data = self.get_data_by_module(self.data_out, None, multiple=True)
        scans_vis = {}
        cur_scan = {}
        for key in module_data:
            scans = module_data[key]
            id, module = key
            scans_vis[id] = {}
            if module in MODULE_SINGLE:
                scans_vis[id] = cur_scan[id] = self.full_dataset(scans, multiple=False)
            else:
                cur_scan[id] = self.full_dataset(scans[curScan], multiple=False)
                for item in scans:
                    scans_vis[id][item] = scans[item] if item != curScan else cur_scan[id]

        super().set_data_by_module(self.data_out, self.module, scans_vis, multiple=True)
        self.load_tabs_data(cur_scan, **kwargs)

    def load_tabs_data(self, scan, **kwargs):
        all = kwargs.get('all', True)
        _compare = kwargs.get('compare', all)
        if _compare:
            self.update_compare_tabs(scan)
        _noise = kwargs.get('noise', all)
        if _noise:
            self.update_noise_tabs(scan)
        _stats = kwargs.get('stats', all)
        if _stats:
            self.update_statistics(scan)

    def update_tabs(self, show_all_default_names=False, show_only_default=False, multiple=True):
        super().update_tabs(show_all_default_names, show_only_default, multiple)

        self.add_compare_tab('Compare')
        self.add_stats_tab('Statistics', header=['Id', 'Applied fit'])
        self.add_noise_stats_tab('Noise', load_contents=False)

    def click_main_tabs(self, index):
        if self.main_tabs.tabText(index) == 'Compare':
            flag = self.load_compare_tab_contents()
            if flag:
                self.load_data(compare=True, all=False)
        elif self.main_tabs.tabText(index) == 'Noise':
            flag = self.load_noise_tab_contents()
            if flag:
                self.load_data(noise=True, all=False)
                self.set_stats_update_flag(True)
        elif self.main_tabs.tabText(index) == 'Statistics':
            if self.get_stats_update_flag():
                self.load_data(stats=True, all=False)
