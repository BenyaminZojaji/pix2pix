from functools import partial
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PySide6.QtWidgets import *
from PySide6.QtUiTools import *
from PySide6.QtCore import *
from PySide6.QtGui import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loader = QUiLoader()
        self.ui = loader.load('Ui/main.ui', None)
        self.ui.show()
        
        self.model = load_model('gray2rgb-2.h5')
        self.fileName = ''

        self.ui.openfile_btn.clicked.connect(self.open)
        self.ui.save_btn.clicked.connect(self.save)
        self.ui.process_btn.clicked.connect(self.process)

    def open(self):
        self.fileName = QFileDialog.getOpenFileName(self, 'Open File')[0]
        img = cv2.imread(self.fileName)
        img = cv2.resize(img, (300, 375))
        img = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.ui.input_label.setPixmap(pixmap)

    def process(self):
        print(self.fileName)
        image = tf.io.read_file(self.fileName)
        image = tf.image.decode_jpeg(image)
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        image = tf.image.resize(image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.expand_dims(image, axis=0)
        pred = self.model(image, training=True)
        pred = pred * 0.5+0.5
        pred = tf.image.convert_image_dtype(pred, tf.uint8)
        pred = tf.squeeze(pred, axis=0)
        pred = np.array(pred)
        #pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        #cv2.imwrite('output.jpg', pred)

        self.pic = pred

        #img = cv2.imread('output.jpg')
        img = cv2.resize(self.pic, (300, 375))
        img = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.ui.output_label.setPixmap(pixmap)

    def save(self):
        file, check = QFileDialog.getSaveFileName(self, 'Save File')
        if check:
            cv2.imwrite(f'{file}.jpg', self.pic)






if __name__=='__main__':
    app = QApplication([])
    window = MainWindow()
    app.exec()