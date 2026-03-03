"""
    1、通过程序将界面运行出来。
    2、实现第一个界面中两种方式的应用功能，无论是图片显示还是摄像头数据显示都需要能将画面实时显示出来，带边界框
    3、实现保存人脸的功能
    4、实现第二个页面应用功能
    5、实现第三个页面的应用功能
"""


import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQT.face_ui import Ui_Form  # 导入你提供的界面类

import os

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()  # 实例化界面类
        self.ui.setupUi(self)  # 设置界面

if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建应用程序对象
    window = MainWindow()  # 创建主窗口
    window.show()  # 显示窗口
    sys.exit(app.exec_())  # 运行应用程序
