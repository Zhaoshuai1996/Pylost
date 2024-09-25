# coding=utf-8
import os

from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from PyQt5.QtCore import QSize, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineSettings, QWebEngineView
from PyQt5.QtWidgets import QSizePolicy as Policy, QTabWidget
from orangewidget.settings import Setting

from pylost_widgets.util.resource_path import resource_path


class OWHelp(OWWidget):
    name = 'Help'
    path = os.path.join(resource_path('..'), 'doc', 'widget_guide', 'index.html')
    path = path.replace(os.path.sep, '/')
    description = 'Documentation of different widgets, available at {}'.format(path)
    icon = "../icons/question.svg"
    priority = 83

    sel_widget = Setting([], schema_only=True)

    widget_list = Setting('')
    want_main_area = 0

    labels = {'Introduction': 'index.html',
              'Data (File)': 'data_file.html',
              'Data (H5)': 'data_h5.html',
              'Data (scans)': 'data_scans.html',
              'Save Data': 'save_data.html',
              'Mask': 'mask.html',
              'Visualize': 'visualize.html',
              'Visualize Compare': 'visualize_compare.html',
              'Visualize Stitch Results': 'visualize_stitch.html',
              'Stitch Parameters': 'stitch_params.html',
              'Fit': 'fit.html',
              'Flip Data': 'flip_data.html',
              'Gravity Correction': 'gravity_correction.html',
              'Integrate Slopes': 'integrate_slopes.html',
              'Interpolate': 'interpolate.html',
              'Optimize XY': 'optimize.html',
              'Rotate': 'rotate.html',
              'Threshold': 'threshold.html',
              'Select Subapertures': 'select_subapertures.html',
              'Operators': 'operators.html',
              'Merge': 'merge.html',
              'Average Scans': 'average_scans.html',
              'Average Subapertues': 'average_subapertures.html',
              'Filter': 'filter.html',
              'Bin Data': 'bin_data.html'}

    def __init__(self):
        super().__init__()

        guid_source = gui.vBox(None, stretch=5)
        browser = QWebEngineView()
        path = os.path.join(resource_path('..'), 'doc', 'source_code', '_build', 'html', 'index.html')
        path = path.replace(os.path.sep, '/')
        if os.path.exists(path):
            html = QUrl(path)
            browser.load(html)
        lbl_path = '<p><a href="{}">Please click here to open this html file in a browser</a></p>'.format(path)
        lbl_source = gui.label(guid_source, self, lbl_path, sizePolicy=(Policy.MinimumExpanding, Policy.Fixed))
        lbl_source.setOpenExternalLinks(True)
        guid_source.layout().addWidget(browser)

        guide = gui.hBox(None)

        lbox = gui.listBox(guide, self, 'sel_widget', 'widget_list', stretch=1, callback=self.select_list_widget,
                           spacing=3)
        lbox.addItems(list(self.labels.keys()))

        vbox = gui.vBox(guide, stretch=5)
        self.lblFile = gui.label(vbox, self, '', sizePolicy=(Policy.MinimumExpanding, Policy.Fixed))
        self.lblFile.setOpenExternalLinks(True)
        self.widget_webview = QWebEngineView()
        vbox.layout().addWidget(self.widget_webview)

        self.widget_webview.settings().setAttribute(QWebEngineSettings.PluginsEnabled, True)
        self.widget_webview.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        self.widget_webview.settings().setAttribute(QWebEngineSettings.FullScreenSupportEnabled, True)
        widget_html = ''
        self.widget_webview.setHtml(widget_html)

        box = gui.vBox(self.controlArea, "Help", stretch=19)
        self.main_tabs = QTabWidget(self)
        box.layout().addWidget(self.main_tabs)
        self.main_tabs.addTab(guide, 'Widget guide')
        self.main_tabs.addTab(guid_source, 'Software documentation')

    def sizeHint(self):
        return QSize(1000, 1000)

    def select_list_widget(self):
        base_path = os.path.join(resource_path('..'), 'doc', 'widget_guide')
        pages = list(self.labels.values())
        path = os.path.join(base_path, pages[self.sel_widget[0]])
        if os.path.exists(path):
            path = path.replace(os.path.sep, '/')
            self.lblFile.setText(
                '<p><a href="{}">Please click here to open this html file in a browser</a></p>'.format(path))
            html = QUrl(path)
            self.widget_webview.load(html)
