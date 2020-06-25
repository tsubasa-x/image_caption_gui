# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/tmp/designerfH8545.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from gtts import gTTS
from pygame import mixer
import tempfile


class Ui_Window(QMainWindow):
    def __init__(self):
        super(Ui_Window, self).__init__()
        self.img_shown = 0
        self.def_font = QtGui.QFont("Noto Sans Mono CJK TC", 10)
        self.setupUi(self)

    def setupUi(self, Window):
        Window.setObjectName("Window")
        Window.setEnabled(True)
        Window.resize(800, 500)
        Window.setFixedSize(QtCore.QSize(800, 500))
        Window.setInputMethodHints(QtCore.Qt.ImhNone)

        self.show_picture = QtWidgets.QLabel(Window)
        self.show_picture.setGeometry(QtCore.QRect(0, 8, 800, 350))
        self.show_picture.setText("")
        self.show_picture.setObjectName("show_picture")
        self.show_picture.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter)

        self.verticalLayoutWidget = QtWidgets.QWidget(Window)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 370, 781, 126))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.line = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        self.verticalLayout.addWidget(self.line)

        self.open_image_button = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.open_image_button.setObjectName("open_image_button")
        self.open_image_button.setFont(self.def_font)

        self.verticalLayout.addWidget(self.open_image_button)

        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

        self.generate_caption_button = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.generate_caption_button.setObjectName("generate_caption_button")
        self.generate_caption_button.setFont(self.def_font)
        self.horizontalLayout_4.addWidget(self.generate_caption_button)

        self.textBrowser = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
        self.textBrowser.setObjectName("textBrowser")

        self.horizontalLayout_4.addWidget(self.textBrowser)
        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.retranslateUi(Window)
        QtCore.QMetaObject.connectSlotsByName(Window)

        self.open_image_button.clicked.connect(self.open_img)
        self.generate_caption_button.clicked.connect(self.generate)

    def retranslateUi(self, Window):
        _translate = QtCore.QCoreApplication.translate
        Window.setWindowTitle(_translate("Window", "Chinese Image Caption"))
        self.open_image_button.setText(_translate("Window", "開啟圖片"))
        self.generate_caption_button.setText(_translate("Window", "生成句子"))

    def open_img(self):
        self.dlg = QFileDialog()
        self.dlg.setNameFilter("Images (*.png *.jpg)")
        # self.dlg.selectNameFilter("Images (*.png *.jpg)")
        # self.img = self.dlg.getOpenFileName()
        if self.dlg.exec_():
            self.img = self.dlg.selectedFiles()
            self.img = QtGui.QPixmap(self.img[0])
            self.scaled_img = self.img.scaled(800, 350, QtCore.Qt.KeepAspectRatio)
            self.show_picture.setPixmap(QtGui.QPixmap(self.scaled_img))
            self.img_shown = 1

    def generate(self):
        if self.img_shown is 0:
            self.textBrowser.setText("請先開啟圖片")
            self.textBrowser.setFont(QtGui.QFont("Noto Sans Mono CJK TC", 12))
        else:
            self.textBrowser.setText("description...    ")
            self.textBrowser.setFont(self.def_font)
            speak('一个穿着泳装的女人躺在海水里')


def speak(text):
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts = gTTS(text=text, lang='zh-tw')
        tts.save('{}.mp3'.format(fp.name))
        mixer.init()
        mixer.music.load('{}.mp3'.format(fp.name))
        mixer.music.play()


def window():
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(80, 80, 80))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)

    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(142, 45, 197).lighter())
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)

    app = QApplication([])
    app.setStyle('Fusion')
    app.setPalette(palette)
    win = Ui_Window()

    win.show()
    sys.exit(app.exec_())


window()
