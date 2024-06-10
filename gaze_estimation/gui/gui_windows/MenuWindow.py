from PyQt5.QtWidgets import QComboBox, QMainWindow, QAction, QApplication, QMessageBox, QPushButton, QVBoxLayout, QWidget, QDesktopWidget
from PyQt5 import QtGui
from gui.gui_windows.TracingWindow import TracingWindow
from gui.gui_windows.WebcamWindow import WebcamWindow

class MenuWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self._camera_identifier = args.camera_identifier
        self._experiment_path = None
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

        self._combo_box = QComboBox(self)
        self._combo_box.addItem('Reproduced L2CS-Net')
        self._combo_box.addItem('Tuned L2CS-Net')
        self._combo_box.addItem('Tuned FastSightNet')
        self._combo_box.currentIndexChanged.connect(self._on_combobox_changed)
        self._on_combobox_changed(self._combo_box.currentIndex())

    def _on_combobox_changed(self, index):
        # Get the selected text
        selected_text = self._combo_box.currentText()
        self._experiment_path = self._get_experiment_path_based_on_selected_model(selected_text)
    
    def _create_menu_layout(self):
        self._menu_layout = QVBoxLayout()
        self._menu_layout.addWidget(self._button_webcam)
        self._menu_layout.addWidget(self._button_tracing)
        self._menu_layout.addWidget(self._combo_box)

    def _add_menu_as_central_widget(self):
        self._central_widget = QWidget()
        self._central_widget.setLayout(self._menu_layout)
        self.setCentralWidget(self._central_widget)

    def _set_window_properties(self):
        self.setWindowTitle('GazeTrack')
        self.setGeometry(0, 0, 300, 500)

    def _open_tracing_window(self):
        print(self._experiment_path)
        self._tracing_window = TracingWindow(self._camera_identifier, self._experiment_path)
        self._tracing_window.windowClosedSignal.connect(self.show)
        self._tracing_window.show()
        self.hide()

    def _open_webcam_window(self):
        print(self._experiment_path)
        self._webcam_window = WebcamWindow(self._camera_identifier, self._experiment_path)
        self._webcam_window.windowClosedSignal.connect(self.show)
        self._webcam_window.show()
        self.hide()

    def _get_experiment_path_based_on_selected_model(self, model_name):
        if model_name == 'Reproduced L2CS-Net':
            return '/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/experiments/ResNet50'
        elif model_name == 'Tuned L2CS-Net':
            return '/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/experiments/Baseline_lr_001'
        else:
            return '/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/experiments/Mobilent_lr_00001_strat_all'

    # def closeEvent(self, event):
    #     self.stopCameraFrameProducer()
    #     self.stopGazeVectorProducer()
    #     self.windowClosedSignal.emit()
    #     event.accept()

