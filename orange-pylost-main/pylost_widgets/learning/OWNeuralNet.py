# coding=utf-8
"""
https://stats.stackexchange.com/questions/7757/data-normalization-and-standardization-in-neural-networks
"""
import collections
import concurrent.futures
import os

import numpy as np
import torch
from Orange.data import Table
from Orange.widgets import gui, widget
from Orange.widgets.utils.concurrent import ThreadExecutor, FutureWatcher, methodinvoke
from Orange.widgets.utils.signals import Output, Input
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt, QThread, pyqtSlot
from PyQt5.QtWidgets import QSizePolicy as Policy, QFileDialog
from functools import partial
from orangewidget.settings import Setting
from orangewidget.widget import Msg
from silx.gui.colors import Colormap
from silx.gui.plot import Plot1D, Plot2D

from pylost_widgets.learning.net.CNNNetwork import CNNNetwork
from pylost_widgets.learning.net.RNNNetwork import RNNNetwork
from pylost_widgets.learning.net.SimpleNetwork import SimpleNetwork
from pylost_widgets.util.Task import Task
from pylost_widgets.util.math import rms


class OWNeuralNet(widget.OWWidget):
    name = 'Neural network'
    description = 'Create and run a neural network model.'
    icon = "../icons/network.svg"
    priority = 1002

    class Inputs:
        data = Input('data', Table)
        data_cnn = Input('cnn_images', dict, auto_summary=False)

    class Outputs:
        data = Output('predictions', Table)

    net_type = Setting('', schema_only=True)
    scale_cnn_to_linear = Setting(1, schema_only=True)
    hidden_size = Setting(30, schema_only=True)
    bidirectional = Setting(False, schema_only=True)
    layer_names = Setting('', schema_only=True)
    layer_sizes = Setting('100', schema_only=True)
    activations = Setting('ReLU', schema_only=True)
    optimizer = Setting('Adam', schema_only=True)
    init_weights = Setting('normal_', schema_only=True)
    scale_inputs = Setting('zscore', schema_only=True)
    loss_fun = Setting('MSELoss', schema_only=True)
    learning_rate = Setting(0.01, schema_only=True)
    shuffle = Setting(False, schema_only=True)
    split_data = Setting(70.0, schema_only=True)
    num_iterations = Setting(300, schema_only=True)
    save_and_load = Setting(True, schema_only=True)
    sel_test = Setting(0, schema_only=True)

    want_main_area = 1

    class Error(widget.OWWidget.Error):
        unknown = Msg("Read error:\n{}")

    def __init__(self):
        super().__init__()
        self.data_in = None
        self.DataStruct = collections.namedtuple('DataStruct', 'X Y')
        self.data_out = None
        self.update_targets_plot = True
        self.fmodel = ''
        self.ytest_m = None

        box = gui.hBox(self.controlArea, "Info", stretch=1)
        self.infolabel = gui.widgetLabel(box, 'No data loaded.', stretch=9)

        self.btnRun = gui.button(box, self, "Run", callback=self.apply, autoDefault=False, stretch=1,
                                 sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        self.btnAbort = gui.button(box, self, 'Abort', callback=self.abort, stretch=1, autoDefault=False,
                                   sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        self.btnAbort.hide()

        box = gui.hBox(self.controlArea, "Save & evaluate", stretch=1)
        self.btnSave = gui.button(box, self, "Save", callback=self.save_model, autoDefault=False, stretch=1,
                                  sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        self.btnEval = gui.button(box, self, "Evaluate", callback=self.evaluate, autoDefault=False, stretch=1,
                                  sizePolicy=Policy(Policy.Fixed, Policy.Fixed))

        nets = ['Simple', 'CNN', 'RNN']
        scales = ['', 'pv', 'zscore', 'tanh']
        gui.comboBox(self.controlArea, self, 'net_type', label='Neural network type', labelWidth=150,
                     orientation=Qt.Horizontal,
                     callback=[self.change_net], sendSelectedValue=True, stretch=1, items=nets,
                     sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        box = gui.vBox(self.controlArea, "Build layers", stretch=1)
        gui.lineEdit(box, self, 'layer_names', label='Layer names', labelWidth=150, orientation=Qt.Horizontal,
                     sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, 'layer_sizes', label='Layer sizes (*)', labelWidth=150, orientation=Qt.Horizontal,
                     sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, 'activations', label='Activations (*)', labelWidth=150, orientation=Qt.Horizontal,
                     sizePolicy=Policy(Policy.Fixed, Policy.Fixed), addSpace=True)
        self.boxCNN = gui.vBox(self.controlArea, "CNN options", stretch=1)
        gui.lineEdit(self.boxCNN, self, 'scale_cnn_to_linear', label='Scale conv2d to linear', labelWidth=150,
                     orientation=Qt.Horizontal, sizePolicy=Policy(Policy.Fixed, Policy.Fixed),
                     tooltip='Typically image size or pooled size to link convolution layers to linear layers')
        self.boxRNN = gui.vBox(self.controlArea, "RNN options", stretch=1)
        gui.lineEdit(self.boxRNN, self, 'hidden_size', label='Hidden size', labelWidth=150, orientation=Qt.Horizontal,
                     sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        gui.checkBox(self.boxRNN, self, 'bidirectional', 'Bi Directional', labelWidth=150,
                     sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(self.controlArea, self, 'optimizer', label='Optimizer', labelWidth=150, orientation=Qt.Horizontal,
                     sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(self.controlArea, self, 'init_weights', label='Weight init method', labelWidth=150,
                     orientation=Qt.Horizontal, sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        gui.comboBox(self.controlArea, self, 'scale_inputs', label='Scale inputs', items=scales, sendSelectedValue=True,
                     labelWidth=150, orientation=Qt.Horizontal, sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(self.controlArea, self, 'loss_fun', label='Loss function', labelWidth=150,
                     orientation=Qt.Horizontal, sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(self.controlArea, self, 'learning_rate', label='Learning rate', labelWidth=150,
                     orientation=Qt.Horizontal, sizePolicy=Policy(Policy.Fixed, Policy.Fixed))

        gui.checkBox(self.controlArea, self, 'shuffle', 'Shuffle', labelWidth=150)
        gui.checkBox(self.controlArea, self, 'save_and_load', 'Save & load model', labelWidth=150)
        gui.lineEdit(self.controlArea, self, 'num_iterations', label='Number of iterations', labelWidth=150,
                     orientation=Qt.Horizontal, sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        gui.hSlider(self.controlArea, self, 'split_data', 'Split to train and test', minValue=0, maxValue=100,
                    step=0.01)
        gui.hSlider(self.controlArea, self, 'sel_test', 'Select test image', minValue=0, maxValue=100, step=1,
                    callback=self.change_test_image)

        self.hideAll()

        box = gui.vBox(self.mainArea, "")
        self.plot_loss = Plot1D()
        self.plot_pred = Plot1D()
        self.plot_test = Plot1D()
        self.plot_params = Plot1D()
        self.plot_pred2d = Plot2D()
        self.plot_test2d = Plot2D()
        self.plot_loss.setGraphTitle('Loss and R2 (pearson)')
        self.plot_pred.setGraphTitle('Train')
        self.plot_test.setGraphTitle('Test / evaluate')
        box.layout().addWidget(self.plot_loss)
        box.layout().addWidget(self.plot_pred)
        box.layout().addWidget(self.plot_test)
        # box.layout().addWidget(self.plot_params)

        self.change_net()
        self._task = None
        self._executor = ThreadExecutor()

    def sizeHint(self):
        return QSize(800, 800)

    @Inputs.data
    def set_data(self, data):
        if data is not None:
            self.data_in = data
            # self.load_data()
        else:
            self.data_in = None
            self.infolabel.setText('No data')

    @Inputs.data_cnn
    def set_data_cnn(self, data):
        if data is not None:
            self.data_in = self.DataStruct(X=data['X'], Y=data['Y'])
            # self.load_data()
        else:
            self.data_in = None
            self.infolabel.setText('No data')

    def abort(self):
        self.btnRun.setEnabled(True)
        self.btnAbort.hide()
        self.info.set_output_summary('Aborted loading')
        if self._task is not None:
            self.cancel()
        self.finProgressBar()

    def cancel(self):
        if self._task is not None:
            self._task.cancel()
            assert self._task.future.done()
            self._task.watcher.done.disconnect(self._task_finished)
            self._task = None

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()

    def hideAll(self):
        self.boxRNN.hide()
        self.boxCNN.hide()

    def change_net(self):
        self.hideAll()
        if self.net_type == 'CNN':
            self.boxCNN.show()
        elif self.net_type == 'RNN':
            self.boxRNN.show()

    def init_data(self):
        data = self.data_in
        # self.data_in = self.DataStruct(X=np.nan_to_num(self.data_in.X), Y=np.nan_to_num(self.data_in.Y))
        self.data_train = self.data_test = data
        if self.split_data > 0:
            n = int((data.X.shape[0]) * self.split_data / 100)
            if isinstance(data, Table):
                self.data_train = data[:n, :]
                self.data_test = data[n:, :]
            else:
                self.data_train = self.DataStruct(X=data.X[:n, :], Y=data.Y[:n, :])
                self.data_test = self.DataStruct(X=data.X[n:, :], Y=data.Y[n:, :])

        self.controls.sel_test.setMaximum(len(self.data_test.X) - 1)

        if self.shuffle:
            if isinstance(self.data_in, Table):
                self.data_train.shuffle()
            else:
                a = np.arange(self.data_train.X.shape[0])
                np.random.shuffle(a)
                self.data_train = self.DataStruct(X=self.data_train.X[a], Y=self.data_train.Y[a])

        if not any(self.layer_sizes):
            raise Exception('Layer sizes cannot be empty')
        else:
            self.layer_sizes_arr = [int(x) for x in self.layer_sizes.split(',') if x != '']
            self.layer_shapes = []
            for i in np.arange(len(self.layer_sizes_arr) + 1):
                if i == 0:
                    self.layer_shapes += [[data.X.shape[1], self.layer_sizes_arr[i]]]
                elif i == len(self.layer_sizes_arr):
                    self.layer_shapes += [[self.layer_sizes_arr[i - 1], data.Y.shape[1] if data.Y.ndim > 1 else 1]]
                else:
                    self.layer_shapes += [[self.layer_sizes_arr[i - 1], self.layer_sizes_arr[i]]]
        if not any(self.activations):
            raise Exception('Activations cannot be empty')
        else:
            self.activations_arr = self.activations.split(',')
            if len(self.activations_arr) == 1:
                self.activations_arr = self.activations_arr * len(self.layer_sizes_arr)
            elif len(self.activations_arr) != len(self.layer_sizes_arr):
                raise Exception('Length of activations is larger than sizes')
            self.activations_arr = [x.split('-') for x in self.activations_arr]

        if self.net_type == 'CNN':
            self.layer_names_arr = [x.split(',') for x in self.layer_names.split('|')]
            if len(self.layer_names_arr) == 1:
                self.layer_names_arr += [['']]
            self.layer_names_arr = [[self.parse_layer_names(y) for y in x] for x in self.layer_names_arr]
        else:
            if not any(self.layer_names):
                self.layer_names_arr = ['Linear'] * len(self.layer_sizes_arr)
            else:
                self.layer_names_arr = self.layer_names.split(',')
                self.layer_names_arr = [self.parse_layer_names(y) for y in self.layer_names_arr]
                if len(self.layer_names_arr) == 1:
                    self.layer_names_arr = self.layer_names_arr * len(self.layer_sizes_arr)
                elif len(self.layer_names_arr) != len(self.layer_sizes_arr):
                    raise Exception('Length of layer names is larger than sizes')

        self.xi = np.arange(self.num_iterations)

    @staticmethod
    def parse_layer_names(name):
        if name == 'c':
            return 'Conv2d'
        elif name == 'l':
            return 'Linear'
        elif name == 'm':
            return 'MaxPool2d'
        return name

    def save_model(self):
        try:
            self.fmodel, _ = QFileDialog.getSaveFileName(self, 'Save network model', self.fmodel, '*.pt')
            if hasattr(self, 'net'):
                torch.save(self.net.state_dict(), self.fmodel)
        except Exception as e:
            self.Error.unknown(repr(e))

    def change_test_image(self):
        if np.any(self.ytest_m):
            s = self.sel_test
            self.update_predictions(self.ytest_m, typ='all', update=True, s=s)
            self.info.set_output_summary('Evaluation: rms={:.4f} \nloss={:.4f}, r2={:.3f}'.format(
                rms(self.ytest_m[s].ravel() - self.data_test.Y[s].ravel()), float(self.loss_test), self.r2_test))

    def evaluate(self):
        try:
            if self.data_in is not None:
                if not os.path.exists(self.fmodel):
                    self.fmodel, _ = QFileDialog.getOpenFileName(self, 'Open network model')
                if os.path.exists(self.fmodel):
                    self.setStatusMessage('')
                    self.clear_messages()
                    self.update_targets_plot = True

                    self.init_data()
                    kwargs = {'init_weights_method': self.init_weights,
                              'scale_inputs': self.scale_inputs,
                              'iterations': self.num_iterations,
                              'loss_fun': self.loss_fun,
                              'optimizer': self.optimizer,
                              'learning_rate': self.learning_rate}
                    if self.net_type == 'CNN':
                        self.net = CNNNetwork(self.layer_names_arr, self.layer_shapes, self.activations_arr,
                                              scale_cnn_to_linear=self.scale_cnn_to_linear, **kwargs)
                    elif self.net_type == 'RNN':
                        self.net = RNNNetwork(self.layer_names_arr, self.layer_shapes, self.activations_arr,
                                              hidden_size=self.hidden_size, bidir=self.bidirectional,
                                              **kwargs)
                    else:
                        self.net = SimpleNetwork(self.layer_names_arr, self.layer_shapes, self.activations_arr,
                                                 **kwargs)

                    self.net.load_state_dict(torch.load(self.fmodel))
                    self.net.eval()
                    self.ytest_m, self.loss_test, self.r2_test = self.net.evaluate(self.data_in.X, self.data_in.Y)
                    self.update_predictions(self.ytest_m, typ='all', update=True)
                    self.info.set_output_summary('Evaluation: rms={:.4f} \nloss={:.4f}, r2={:.3f}'.format(
                        rms(self.ytest_m.ravel() - self.data_in.Y.ravel()), float(self.loss_test), self.r2_test))
        except Exception as e:
            self.Error.unknown(repr(e))

    def apply(self):
        try:
            if self.data_in is not None:
                if self._task is not None:
                    self.cancel()
                assert self._task is None
                self.setStatusMessage('')
                self.clear_messages()
                self.update_targets_plot = True

                self.init_data()
                kwargs = {'init_weights_method': self.init_weights,
                          'scale_inputs': self.scale_inputs,
                          'iterations': self.num_iterations,
                          'loss_fun': self.loss_fun,
                          'optimizer': self.optimizer,
                          'learning_rate': self.learning_rate}
                if self.net_type == 'CNN':
                    self.net = CNNNetwork(self.layer_names_arr, self.layer_shapes, self.activations_arr,
                                          scale_cnn_to_linear=self.scale_cnn_to_linear, **kwargs)
                elif self.net_type == 'RNN':
                    self.net = RNNNetwork(self.layer_names_arr, self.layer_shapes, self.activations_arr,
                                          hidden_size=self.hidden_size, bidir=self.bidirectional,
                                          **kwargs)
                else:
                    self.net = SimpleNetwork(self.layer_names_arr, self.layer_shapes, self.activations_arr, **kwargs)
                # self.writer = self.net.get_tensorboard()
                self.net.progress.connect(self.report_progress)
                self.net.sigLoss.connect(self.update_loss)
                self.net.sigR2.connect(self.update_r2)
                self.net.sigPredictions.connect(self.update_predictions)
                self.net.sigParams.connect(self.update_params)
                self._task = task = Task()
                end_progressbar = methodinvoke(self, "finProgressBar", ())

                def callback():
                    if task.cancelled:
                        end_progressbar()
                        raise Exception('Aborted loading')

                load_fun = partial(self.net.run_net, callback=callback, Xa=self.data_train.X, Ya=self.data_train.Y,
                                   save_and_load=self.save_and_load)

                self.startProgressBar()
                task.future = self._executor.submit(load_fun)
                task.watcher = FutureWatcher(task.future)
                task.watcher.done.connect(self._task_finished)
                self.btnRun.setEnabled(False)
                self.btnAbort.show()
        except Exception as e:
            self.Error.unknown(repr(e))

    @pyqtSlot(concurrent.futures.Future)
    def _task_finished(self, f):
        assert self.thread() is QThread.currentThread()
        assert self._task is not None
        assert self._task.future is f
        assert f.done()

        self._task = None
        self.endProgressBar()
        try:
            self.btnRun.setEnabled(True)
            self.btnAbort.hide()

            ytrain_m, loss_train, r2_train = f.result()
            if np.any(ytrain_m):
                self.ytest_m, self.loss_test, self.r2_test = self.net.evaluate(self.data_test.X, self.data_test.Y)
                self.info.set_output_summary(
                    'Finished training: train rms={:.4f}, test rms={:.4f} \ntrain loss={:.4f}, test loss={:.4f}, train_r2={:.3f}, test_r2={:.3f}'.format(
                        rms(ytrain_m.ravel() - self.data_train.Y.ravel()),
                        rms(self.ytest_m.ravel() - self.data_test.Y.ravel()),
                        float(loss_train), float(self.loss_test), r2_train, self.r2_test))
                self.update_predictions(self.ytest_m, typ='test', update=True)
                self.Outputs.data.send(self.ytest_m)

        except Exception as e:
            return self.Error.unknown(repr(e))

    def report_progress(self, val):
        try:
            self.setProgressValue(val)
        except Exception as e:
            self.Error.unknown(repr(e))

    @pyqtSlot(float)
    def setProgressValue(self, value):
        assert self.thread() is QThread.currentThread()
        self.progressBarSet(value)

    @pyqtSlot()
    def finProgressBar(self):
        assert self.thread() is QThread.currentThread()
        self.endProgressBar()

    def startProgressBar(self):
        try:
            self.progressBarInit()
        except Exception as e:
            self.Error.unknown(repr(e))

    def endProgressBar(self):
        try:
            self.progressBarFinished()
        except Exception as e:
            self.Error.unknown(repr(e))

    def update_loss(self, y):
        self.plot_loss.addCurve(self.xi, y, legend='Loss')

    def update_r2(self, y):
        self.plot_loss.addCurve(self.xi, y, legend='R2', yaxis='right')

    def update_predictions(self, y, typ='train', dim=2, update=True, s=-1):
        if s < 0:
            s = np.random.randint(0, len(y))
        if typ == 'train':
            plot = self.plot_pred2d if dim == 2 else self.plot_pred
            data = self.data_train
        elif typ == 'test':
            plot = self.plot_test2d if dim == 2 else self.plot_test
            data = self.data_test
        else:
            plot = self.plot_test2d if dim == 2 else self.plot_test
            data = self.data_in
        if dim == 2:
            arr0 = np.asarray(data.Y[s, 0])
            arr1 = np.asarray(data.X[s, 0])
            arr = np.asarray(y)[s, 0]
            plot.show()
            plot.addImage(arr0, legend='target', colormap=Colormap('turbo'))
            plot.addImage(arr, legend='pred', colormap=Colormap('turbo'), origin=(0, arr0.shape[0] + 20))
            plot.addImage(arr1, legend='input X', colormap=Colormap('turbo'), origin=(0, 2 * (arr0.shape[0] + 20)))
        else:
            yi = np.arange(len(y[s, :]))
            yq = yi  # np.argsort(y)
            arr0 = np.asarray(data.Y[s, :].ravel())
            plot.addCurve(yi, arr0[yq], legend='Targets', z=100)
            arr = np.asarray(y)[s, 0].ravel()
            plot.addCurve(yi, arr[yq], legend='Predictions')

    def update_params(self, y):
        total = []
        for i, a in enumerate(y):
            total += list(a.ravel())
            # self.plot_params.addCurve(np.arange(a.size), a.ravel(), legend='{}'.format(i))
            h = np.histogram(a.ravel(), bins=100)
            self.plot_params.addHistogram(h[0], h[1], legend='{}'.format(i))
        # h = np.histogram(total, bins=100)
        # self.plot_params.addHistogram(h[0], h[1], legend='all')
