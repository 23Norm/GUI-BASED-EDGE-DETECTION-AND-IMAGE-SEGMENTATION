import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage

class WebcamHandler:
    def __init__(self, display_label: QLabel):
        self.display_label = display_label
        self.cap = None
        self.timer = QTimer()
        self.current_frame = None  # Store the latest frame

    def startWebcam(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)

    def updateFrame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame  # Update the current frame
            self.displayImage(frame)

    def displayImage(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.display_label.setPixmap(pixmap)
        self.display_label.setScaledContents(True)

    def stopWebcam(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.cap = None  # Reset the cap
    
    def getCurrentFrame(self):
        return self.current_frame
