import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QApplication, QMessageBox, QPushButton, QVBoxLayout, QWidget
from PyQt5 import QtGui, QtCore
from gui.gui_windows.TracingWindow import TracingWindow
from gui.gui_windows.MenuWindow import MenuWindow
import os

def main():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)
    app = QApplication(sys.argv)
    menu_window = MenuWindow()
    menu_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()