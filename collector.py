import os
import shutil
import sys
import time
import threading
import numpy as np

import cv2

from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLabel, QPushButton,
    QInputDialog, QMessageBox, QLineEdit, QRadioButton, QFileDialog,
    QWidget, QAction, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox)
from PyQt5.QtGui import QFont, QPainter, QImage, QTextCursor, QPixmap

try:
    import Queue as Queue
except:
    import queue as Queue


IMG_SIZE    = 640,480           # 640,480 or 1280,720 or 1920,1080
IMG_FORMAT  = QImage.Format_RGB888
DISP_SCALE  = 2                 # Scaling factor for display image
DISP_MSEC   = 50                # Delay between display cycles
CAP_API     = cv2.CAP_ANY       # API: CAP_ANY or CAP_DSHOW etc...
EXPOSURE    = 0                 # Zero for automatic exposure
TEXT_FONT   = QFont("Courier", 10)

camera_num  = 1                 # Default camera (first in list)
image_queue = Queue.Queue()     # Queue to hold images
capturing   = True              # Flag to indicate capturing
image_index = 0
save_dir = None
save_images = False

# Grab images from the camera (separate thread to control FPS)
def grab_images(cam_num, queue):
    global image_index, save_dir, save_images
    
    # Configure depth and color streams
    cap = cv2.VideoCapture(cam_num-1)

    while capturing:
        if cap.grab():
            retval, color_image = cap.read()  # cap.retrieve(0)
            if color_image is not None and queue.qsize() < 2:
                color_image = cv2.flip(color_image, 1)
                color_image_save = color_image.copy()
                queue.put(color_image)
                # Store images
                if save_images:
                    cv2.imwrite(
                        "./{}/rgb_{:06d}.png".format(save_dir, image_index),
                        cv2.resize(color_image, (224, 224))
                    )
                    image_index += 1
            else:
                time.sleep(DISP_MSEC / 1000.0)
        else:
            print("Error: can't grab camera image")
            break

    # Stop streaming
    cap.release()

# Image widget
class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()

# Main window
class MainVideoCapture(QMainWindow):

    # Create main window
    def __init__(self, personInfo, fileName, parent=None):
        QMainWindow.__init__(self, parent)

        # data specifics
        self.participantInformation = personInfo  # '200228_kuh_150'
        print("Participant:", self.participantInformation)
        
        self.textToDisplay = open(fileName, encoding='utf-8').readlines()
        self.textIndex = 0
        self.maxTextIndex = len(self.textToDisplay) - 1

        text_save_dir = 'RawData' + '/' + self.participantInformation
        if not os.path.exists(text_save_dir):
            os.makedirs(text_save_dir)
        file_name = text_save_dir + '/' + 'gt.txt'
        text_to_write = open(file_name, 'w')
        text_to_write.writelines(self.textToDisplay)

        self.central = QWidget(self)
        print("Camera number %u" % camera_num)
        print("Image size %u x %u" % IMG_SIZE)
        if DISP_SCALE > 1:
            print("Display scale %u:1" % DISP_SCALE)

        self.vlayout = QVBoxLayout()        # Window layout
        self.displays = QHBoxLayout()
        
        self.buttonGroup = QVBoxLayout()
        self.buttonSubGroup = QVBoxLayout()
        
        self.buttonStart = QPushButton('Start')
        self.buttonStart.clicked.connect(self.startDataCollection)
        self.buttonStop = QPushButton('Stop')
        self.buttonStop.clicked.connect(self.stopDataCollection)
        self.buttonNext = QPushButton('Next')
        self.buttonNext.clicked.connect(self.nextDataCollection)
        self.buttonRedo = QPushButton('Redo')
        self.buttonRedo.clicked.connect(self.redoDataCollection)
        
        self.buttonSubGroup.addWidget(self.buttonStart, 1)
        self.buttonSubGroup.addWidget(self.buttonStop, 1)
        self.buttonSubGroup.addWidget(self.buttonNext, 1)
        self.buttonSubGroup.addWidget(self.buttonRedo, 1)

        self.buttonGroup.addLayout(self.buttonSubGroup, 1)
 
        self.textForTyping = QLabel(self.textToDisplay[self.textIndex], self)
        self.textForTyping.setFont(QFont('Arial', 36))
        self.textForTyping.setAlignment(Qt.AlignCenter)
        self.disp = ImageWidget(self)
        self.displays.addWidget(self.disp, 2)
        self.displays.addLayout(self.buttonGroup, 1)
        self.vlayout.addWidget(self.textForTyping, 4)
        self.vlayout.addLayout(self.displays)
        self.vlayout.setAlignment(Qt.AlignCenter)

        self.central.setLayout(self.vlayout)
        self.setCentralWidget(self.central)

        self.mainMenu = self.menuBar()      # Menu bar
        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.triggered.connect(self.close)
        self.fileMenu = self.mainMenu.addMenu('&File')
        self.fileMenu.addAction(exitAction)

    # Start image capture & display
    def start(self):
        self.timer = QTimer(self)           # Timer to trigger display
        self.timer.timeout.connect(lambda: 
                    self.show_image(image_queue, self.disp, DISP_SCALE))
        self.timer.start(DISP_MSEC)         
        self.capture_thread = threading.Thread(target=grab_images, 
                    args=(camera_num, image_queue))
        self.capture_thread.start()         # Thread to grab images

    # Fetch camera image from queue, and display it
    def show_image(self, imageq, display, scale):
        if not imageq.empty():
            image = imageq.get()
            if image is not None and len(image) > 0:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.display_image(img, display, scale)

    # Display an image, reduce size if required
    def display_image(self, img, display, scale=1):
        disp_size = img.shape[1]//scale, img.shape[0]//scale
        disp_bpl = disp_size[0] * 3
        if scale > 1:
            img = cv2.resize(img, disp_size, 
                             interpolation=cv2.INTER_CUBIC)
        qimg = QImage(img.data, disp_size[0], disp_size[1], 
                      disp_bpl, IMG_FORMAT)
        display.setImage(qimg)

    # Window is closing: stop video capture
    def closeEvent(self, event):
        global capturing
        capturing = False
        self.capture_thread.join()
    
    def startDataCollection(self):
        global save_dir
        save_dir = 'RawData' + '/' + self.participantInformation + '/' + str(self.textIndex)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            global save_images
            save_images = True
            
            if self.textIndex + 1 % 10 == 1:
                information_text = '[Start] data collection starts for {}-st text...'.format(self.textIndex+1)
            elif self.textIndex + 1 % 10 == 2:
                information_text = '[Start] data collection starts for {}-nd text...'.format(self.textIndex+1)
            elif self.textIndex + 1 % 10 == 3:
                information_text = '[Start] data collection starts for {}-rd text...'.format(self.textIndex+1)
            else:
                information_text = '[Start] data collection starts for {}-th text...'.format(self.textIndex+1)
            print(information_text)
        else:
            QMessageBox.question(
                self, 'Alert Message','You have already collected the current sentence!', QMessageBox.Ok)

    def stopDataCollection(self):
        global image_index, save_dir, save_images
        image_index = 0
        save_images = False
        save_dir = None
        print('\t data collection stopped...')

    def nextDataCollection(self):
        save_dir_check = 'RawData' + '/' + self.participantInformation + '/' + str(self.textIndex)
        if not os.path.exists(save_dir_check):
            QMessageBox.question(
                self, 'Alert Message', 'You have not collected the current sentence!', QMessageBox.Ok)
            return
        
        global save_images
        if save_images:
            self.stopDataCollection()

        if self.textIndex < self.maxTextIndex:
            self.textIndex += 1
            if self.textIndex % 10 == 1:
                information_text = '{}-st text collected'.format(self.textIndex)
            elif self.textIndex % 10 == 2:
                information_text = '{}-nd text collected'.format(self.textIndex)
            elif self.textIndex % 10 == 3:
                information_text = '{}-rd text collected'.format(self.textIndex)
            else:
                information_text = '{}-th text collected'.format(self.textIndex)
            reply = QMessageBox.information(self, 'Information', information_text, QMessageBox.Ok)
            if reply == QMessageBox.Ok:
                self.textForTyping.setText(self.textToDisplay[self.textIndex])
        else:
            reply = QMessageBox.information(
                self, 'Final Message', 'Data Collection Finished!', QMessageBox.Ok)
            if reply == QMessageBox.Ok:
                self.close()

    def redoDataCollection(self):
        save_dir_check = 'RawData' + '/' + self.participantInformation + '/' + str(self.textIndex)
        if not os.path.exists(save_dir_check):
            QMessageBox.question(self, 'Alert Message', 'Nothing to redo!', QMessageBox.Ok)
            return
        shutil.rmtree(save_dir_check)
        print('\t redo data collection...')
        self.startDataCollection()

# Questionnaire window to collect basic user information
class Questionnaire(QWidget):
    switch_window = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)
        
        self.hboxRadioButtons = QHBoxLayout()
        gboxRadioButtons = QGroupBox()
        for sex in ['Male', 'Female', 'Not to reveal']:
            buttonSex = QRadioButton(sex)
            self.hboxRadioButtons.addWidget(buttonSex)
        gboxRadioButtons.setLayout(self.hboxRadioButtons)
        gboxRadioButtons.setStyleSheet(
            "QGroupBox { background-color: rgb(236, 236, 236); border: 1px solid rgb(236, 236, 236);}")

        gboxNextCancelButtons = QGroupBox()
        hboxNextCancelButtons = QHBoxLayout()
        self.cancelButton = QPushButton('Cancel')
        self.cancelButton.clicked.connect(self.handleCancel)
        self.nextButton = QPushButton('Next')
        self.nextButton.clicked.connect(self.handleNext)
        hboxNextCancelButtons.addWidget(self.cancelButton)
        hboxNextCancelButtons.addWidget(self.nextButton)
        gboxNextCancelButtons.setLayout(hboxNextCancelButtons)
        gboxNextCancelButtons.setStyleSheet(
            "QGroupBox { background-color: rgb(236, 236, 236); border: 1px solid rgb(236, 236, 236);}")

        grid.addWidget(QLabel('Initial:'), 0, 0)
        grid.addWidget(QLabel('Age:'), 1, 0)
        grid.addWidget(QLabel('Gender:'), 2, 0)
        grid.addWidget(QLabel('Data Type:'), 3, 0)

        self.personInitial = QLineEdit()
        self.personAge = QLineEdit()
        self.personTypingSpeed = QLineEdit()
        grid.addWidget(self.personInitial, 0, 1)
        grid.addWidget(self.personAge, 1, 1)
        grid.addWidget(gboxRadioButtons, 2, 1)
        grid.addWidget(self.personTypingSpeed, 3, 1)

        grid.addWidget(gboxNextCancelButtons, 4, 0, 1, 2)

        self.setWindowTitle('Questionnaire')
    
    def handleCancel(self):
        self.close()
    
    def handleNext(self):
        sex = [self.hboxRadioButtons.itemAt(i).widget().text() for i in range(3) if self.hboxRadioButtons.itemAt(i).widget().isChecked()]
        if len(sex) == 0:
            QMessageBox.question(self, 'Alert Message', 'Select Gender!', QMessageBox.Ok)
            return
        
        emitText = "_".join(
            [self.personInitial.text(), sex[0], self.personAge.text(), self.personTypingSpeed.text()])
        
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select Text to Write",
            "",
            "All Files (*);;Text Files (*.txt)",
            options=options
        )

        self.switch_window.emit(emitText, fileName)

# Controller for switching between windows
class Controller:
    def __init__(self):
        pass

    def showQuestionnaire(self):
        self.questionnaire = Questionnaire()
        self.questionnaire.switch_window.connect(self.showVideoCapture)
        self.questionnaire.show()
    
    def showVideoCapture(self, text, fileName):
        self.questionnaire.close()

        if fileName == "":
            print("No file selected")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle('No file selected')
            msg.setText("No file selected.")
            msg.setInformativeText("You need to select a text file to continue...")
            msg.exec_()
        else:
            self.videoCapture = MainVideoCapture(text, fileName)
            self.videoCapture.show()
            self.videoCapture.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    controller = Controller()
    controller.showQuestionnaire()
    sys.exit(app.exec_())
