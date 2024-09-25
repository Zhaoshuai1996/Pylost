# coding=utf-8
'''
Created on Mar 29, 2018

Animation of a wheel when a process is running in the background for a long time e.g. converting raw format to h5

@author: ADAPA
'''
import math

from PyQt5.Qt import Qt
from PyQt5.QtGui import QPalette, QPainter, QBrush, QColor, QPen
from PyQt5.QtWidgets import QWidget


class Overlay(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        palette = QPalette(self.palette())
        palette.setColor(palette.Background, Qt.transparent)
        self.setPalette(palette)

    def paintEvent(self, event):
        painter = QPainter(self)
        # painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), QBrush(QColor(255, 255, 255, 127)))
        painter.setPen(QPen(Qt.NoPen))

        for i in range(6):
            if (self.counter / 5) % 6 == i:
                painter.setBrush(QBrush(QColor(127 + (self.counter % 5) * 32, 127, 127)))
            else:
                painter.setBrush(QBrush(QColor(127, 127, 127)))
                painter.drawEllipse(
                    self.width() / 2 + 30 * math.cos(2 * math.pi * i / 6.0) - 10,
                    self.height() / 2 + 30 * math.sin(2 * math.pi * i / 6.0) - 10,
                    20, 20)

                # painter.end()

    def showEvent(self, event):
        self.timer = self.startTimer(50)
        self.counter = 0
        self.show_flag = 1

    def timerEvent(self, event):
        self.counter += 1
        self.update()
        #         if self.counter == 60:
        #             self.killTimer(self.timer)
        #             self.hide()
        if not self.show_flag:
            self.killTimer(self.timer)
            self.hide()

    def stop(self):
        self.show_flag = 0
