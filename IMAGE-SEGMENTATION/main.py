import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QSlider, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from sklearn.cluster import KMeans
from webcam import WebcamHandler  # Import the WebcamHandler class

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Edge Detection and Image Segmentation')
        self.setGeometry(100, 100, 1200, 800)

        # Widgets for file loading and image display
        self.image_label = QLabel(self)
        self.result_label = QLabel(self)

        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.loadImage)

        # Sobel Edge Detection
        self.sobel_button = QPushButton('Apply Sobel', self)
        self.sobel_button.clicked.connect(self.applySobel)

        # Canny Edge Detection
        self.canny_button = QPushButton('Apply Canny', self)
        self.canny_button.clicked.connect(self.applyCanny)

        self.canny_slider1 = QSlider(Qt.Horizontal, self)
        self.canny_slider1.setRange(0, 255)
        self.canny_slider1.setValue(100)
        self.canny_slider1.setTickPosition(QSlider.TicksBelow)

        self.canny_slider2 = QSlider(Qt.Horizontal, self)
        self.canny_slider2.setRange(0, 255)
        self.canny_slider2.setValue(200)
        self.canny_slider2.setTickPosition(QSlider.TicksBelow)

        # Thresholding Segmentation
        self.threshold_button = QPushButton('Apply Threshold', self)
        self.threshold_button.clicked.connect(self.applyThreshold)

        self.threshold_slider = QSlider(Qt.Horizontal, self)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(127)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)

        # K-Means Clustering Segmentation
        self.kmeans_button = QPushButton('Apply K-Means', self)
        self.kmeans_button.clicked.connect(self.applyKMeans)

        self.kmeans_slider = QSlider(Qt.Horizontal, self)
        self.kmeans_slider.setRange(1, 10)
        self.kmeans_slider.setValue(3)
        self.kmeans_slider.setTickPosition(QSlider.TicksBelow)

        # Webcam Functionality
        self.start_camera_button = QPushButton('Start Camera', self)
        self.start_camera_button.clicked.connect(self.startCamera)

        self.stop_camera_button = QPushButton('Stop Camera', self)
        self.stop_camera_button.clicked.connect(self.stopCamera)

        self.capture_button = QPushButton('Capture Image', self)
        self.capture_button.clicked.connect(self.captureImage)

        # Layout
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.result_label)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.sobel_button)
        controls_layout.addWidget(self.canny_button)
        controls_layout.addWidget(self.canny_slider1)
        controls_layout.addWidget(self.canny_slider2)
        controls_layout.addWidget(self.threshold_button)
        controls_layout.addWidget(self.threshold_slider)
        controls_layout.addWidget(self.kmeans_button)
        controls_layout.addWidget(self.kmeans_slider)
        controls_layout.addWidget(self.start_camera_button)
        controls_layout.addWidget(self.stop_camera_button)
        controls_layout.addWidget(self.capture_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(controls_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Webcam handler
        self.webcam_handler = WebcamHandler(self.image_label)

    def loadImage(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Image Files (*.jpg *.png)", options=options)
        if file_name:
            self.image = cv2.imread(file_name)
            self.displayImage(self.image, self.image_label)

    def displayImage(self, image, label, fixed_width = 800, fixed_height = 600):
        # Convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fixed width and height
        image = cv2.resize(image, (fixed_width, fixed_height))
        
        # Convert image to QImage
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(q_img)
        
        # Set QPixmap to label
        label.setPixmap(pixmap)
        label.setScaledContents(True)


    def applySobel(self):
        image = self.getCurrentImage()
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = np.uint8(sobel)
            self.displayImage(sobel, self.result_label)

    def applyCanny(self):
        image = self.getCurrentImage()
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold1 = self.canny_slider1.value()
            threshold2 = self.canny_slider2.value()
            edges = cv2.Canny(gray, threshold1, threshold2)
            self.displayImage(edges, self.result_label)

    def applyThreshold(self):
        image = self.getCurrentImage()
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold_value = self.threshold_slider.value()
            _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            self.displayImage(thresh, self.result_label)

    def applyKMeans(self):
        image = self.getCurrentImage()
        if image is not None:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixel_values = img.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)

            k = self.kmeans_slider.value()
            _, labels, centers = cv2.kmeans(pixel_values, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented_image = centers[labels.flatten()]
            segmented_image = segmented_image.reshape(img.shape)

            self.displayImage(segmented_image, self.result_label)

    def startCamera(self):
        self.webcam_handler.startWebcam()

    def stopCamera(self):
        self.webcam_handler.stopWebcam()

    def captureImage(self):
        frame = self.webcam_handler.getCurrentFrame()
        if frame is not None:
            self.image = frame.copy()  # Capture the current frame
            self.displayImage(self.image, self.image_label)

    def getCurrentImage(self):
        if hasattr(self, 'image'):
            return self.image
        return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    processor = ImageProcessor()
    processor.show()
    sys.exit(app.exec_())
