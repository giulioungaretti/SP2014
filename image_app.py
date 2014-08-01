#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ZetCode PyQt4 tutorial

In this example, we dispay an image
on the window.

author: Jan Bodnar
website: zetcode.com
last edited: September 2011
"""

import sys
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg


class app(QtGui.QWidget):

    def __init__(self):
        super(app, self).__init__()

        self.initUI()

    def initUI(self):

        hbox = QtGui.QHBoxLayout(self)
        self.imv1 = pg.ImageView(view=pg.PlotItem(title='Pilatus image'))
        hbox.addWidget(self.imv1)
        self.setLayout(hbox)
        self.move(300, 200)
        self.setWindowTitle('Red Rock')
        self.show()


def main():
    start_app = QtGui.QApplication(sys.argv)
    win = app()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
