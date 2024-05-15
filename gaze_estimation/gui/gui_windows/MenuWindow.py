from PyQt5.QtWidgets import QMainWindow, QAction, QApplication, QMessageBox, QPushButton, QVBoxLayout, QWidget, QDesktopWidget
from PyQt5 import QtGui
from gui.gui_windows.TracingWindow import TracingWindow
from gui.gui_windows.WebcamWindow import WebcamWindow

class MenuWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._init_widgets()
        self._create_menu_layout()
        self._add_menu_as_central_widget()
        self._set_window_properties()
        self.move_to_primary_screen()

    def move_to_primary_screen(self):
        screen = QDesktopWidget().screenGeometry()
        self.move(screen.left(), screen.top())

    def _init_widgets(self):
        self._button_webcam = QPushButton('webcam', self)
        self._button_webcam.clicked.connect(self._open_webcam_window)
        self._button_tracing = QPushButton('tracing', self)
        self._button_tracing.clicked.connect(self._open_tracing_window)
    
    def _create_menu_layout(self):
        self._menu_layout = QVBoxLayout()
        self._menu_layout.addWidget(self._button_webcam)
        self._menu_layout.addWidget(self._button_tracing)

    def _add_menu_as_central_widget(self):
        self._central_widget = QWidget()
        self._central_widget.setLayout(self._menu_layout)
        self.setCentralWidget(self._central_widget)

    def _set_window_properties(self):
        self.setWindowTitle('App name')
        self.setGeometry(0, 0, 300, 500)

    def _open_tracing_window(self):
        self._tracing_window = TracingWindow()
        self._tracing_window.windowClosedSignal.connect(self.show)
        self._tracing_window.show()
        self.hide()

    def _open_webcam_window(self):
        self._webcam_window = WebcamWindow()
        self._webcam_window.windowClosedSignal.connect(self.show)
        self._webcam_window.show()
        self.hide()