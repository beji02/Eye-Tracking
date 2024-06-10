# import sys
# from PyQt5.QtCore import QThread, pyqtSignal, QTimer
# from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
# import time

# class Worker(QThread):
#     message_sent = pyqtSignal(int)
#     finished = pyqtSignal()

#     def __init__(self):
#         super().__init__()

#     def send_message(self):
#         self.count += 1
#         # time.sleep(3)
#         print(f"Sending message {self.count}...")
#         self.message_sent.emit(self.count)
#         if self.count == 10:
#             self.timer.stop()
#             self.finished.emit()

#     def run(self):
#         print("Running...")
#         self.count = 0
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.send_message)
#         self.timer.start(3000)  # Start the timer with an interval of 1000 milliseconds (1 second)
#         self.exec()

# class MyWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.init_ui()
#         self.thread = None

#     def init_ui(self):
#         self.start_button = QPushButton('Start')
#         self.stop_button = QPushButton('Stop')

#         self.start_button.clicked.connect(self.start_thread)
#         self.stop_button.clicked.connect(self.stop_thread)

#         layout = QVBoxLayout()
#         layout.addWidget(self.start_button)
#         layout.addWidget(self.stop_button)
#         self.setLayout(layout)

#         self.setWindowTitle('QThread Example')
#         self.show()

#     def start_thread(self):
#         if self.thread is None or not self.thread.isRunning():
#             self.thread = Worker()
#             self.thread.message_sent.connect(self.on_message_sent)
#             self.thread.finished.connect(self.on_thread_finished)
#             self.thread.start()

#     def stop_thread(self):
#         if self.thread is not None and self.thread.isRunning():
#             self.thread.quit()  # Terminate the thread
#             self.thread.wait()  # Wait for the thread to finish
#             self.thread = None
#             print("Thread stopped.")

#     def on_message_sent(self, count):
#         print(f"Received message {count}")

#     def on_thread_finished(self):
#         self.thread = None
#         print("Thread finished.")

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = MyWindow()
#     sys.exit(app.exec_())
