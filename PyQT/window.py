import sys
from PyQt5.QtWidgets import QApplication, QWidget
from face_ui import Ui_Form  # 导入你提供的界面类

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
