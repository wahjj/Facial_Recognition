import sys
import os
import cv2
import torch
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QFileDialog, QTableWidgetItem
from PyQt5.QtCore import QTimer, Qt, QByteArray
from PyQt5.QtGui import QImage, QPixmap
from PyQT.face_ui import Ui_Form
from Face_main.sqlite_db import MySqlite
from Face_main.facenet_model import FaceNet

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # 初始化数据库
        self.db = MySqlite()

        # 初始化摄像头相关变量
        self.cap = None
        self.timer = QTimer()
        self.current_image = None
        self.current_faces = []

        self.yolo_path = r'C:\Users\Lenovo\Desktop\HQYJ\Facial_Recognition\my_yolo\runs\detect\train\weights\best.pt'
        self.face_net = FaceNet(self.yolo_path)

        # 绑定信号槽
        self.init_signals()

        # 初始化数据库表
        self.create_table_if_not_exists()

    def init_signals(self):
        """初始化信号槽连接"""
        # 左侧功能按钮
        self.ui.pushButton_sb_3.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(0))
        self.ui.pushButton_25.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(1))
        self.ui.pushButton_gl_3.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(2))
        self.ui.pushButton_lr_3.clicked.connect(self.close)

        # 第一页：人脸信息录入
        self.ui.pushButton_14.clicked.connect(self.upload_image_for_enrollment)
        self.ui.pushButton_15.clicked.connect(self.start_camera_enrollment)
        self.ui.pushButton_16.clicked.connect(self.capture_face_enrollment)
        self.ui.pushButton_17.clicked.connect(self.stop_camera_enrollment)
        self.ui.pushButton_26.clicked.connect(self.save_face_info)

        # 第二页：人脸识别
        self.ui.pushButton_19.clicked.connect(self.upload_image_for_recognition)
        self.ui.pushButton_20.clicked.connect(self.start_camera_recognition)
        self.ui.pushButton_21.clicked.connect(self.stop_camera_recognition)

        # 第三页：数据库管理
        self.ui.pushButton_22.clicked.connect(self.search_database)
        self.ui.pushButton_23.clicked.connect(self.delete_record)
        self.ui.pushButton_24.clicked.connect(self.refresh_table)

        # 启动定时器用于摄像头显示
        self.timer.timeout.connect(self.update_frame)

    def create_table_if_not_exists(self):
        """创建数据库表"""
        sql = """
        CREATE TABLE IF NOT EXISTS student_info(
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            姓名 TEXT,
            年龄 INTEGER,
            性别 TEXT,
            学号 TEXT,
            录入时间 TEXT,
            照片 BLOB
        )
        """
        result = self.db.operation_sql(sql)
        if result is True:
            print("数据库表创建成功")
        else:
            print(f"数据库表创建失败: {result}")

    def upload_image_for_enrollment(self):
        """上传图片进行人脸录入"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            self.ui.lineEdit_7.setText(file_path)
            self.detect_face_in_image(file_path)
        pass

    def detect_face_in_image(self, image_path):
        """检测图片中的人脸"""
        try:
            face_imgs = self.face_net.getfacepos(image_path)
            if face_imgs:
                self.current_faces = face_imgs
                # 显示第一张检测到的人脸
                self.display_face_image(face_imgs[0])
            else:
                QMessageBox.warning(self, "警告", "未检测到人脸！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"人脸检测失败: {str(e)}")
        pass

    def start_camera_enrollment(self):
        """开启摄像头进行人脸录入"""
        pass

    def capture_face_enrollment(self):
        """拍照进行人脸录入"""
        pass

    def stop_camera_enrollment(self):
        """关闭摄像头"""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        # 清空显示
        self.ui.label_15.clear()
        self.ui.label_30.clear()

    def update_frame(self):
        """更新摄像头帧"""
        pass

    def detect_and_draw_faces(self, frame):
        """检测人脸并绘制边界框"""
        pass

    def save_face_info(self):
        """保存人脸信息到数据库"""
        pass

    def upload_image_for_recognition(self):
        """上传图片进行人脸识别"""
        pass
    def recognize_face_in_image(self, image_path):
        """识别人脸图片"""
        pass

    def find_matches(self, query_embedding, threshold=0.6):
        """查找数据库中最相似的人脸"""
        pass

    def start_camera_recognition(self):
        """开启摄像头进行人脸识别"""
        pass

    def recognize_from_camera(self):
        """从摄像头进行人脸识别"""
        pass

    def recognize_face_in_frame(self, frame):
        """识别人脸帧"""
        pass

    def stop_camera_recognition(self):
        """关闭摄像头识别"""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        # 清空显示
        self.ui.label_24.clear()
        self.ui.plainTextEdit_2.clear()

    def search_database(self):
        """搜索数据库"""
        pass

    def load_table_data(self):
        """加载表格数据"""
        pass

    def delete_record(self):
        """删除记录"""
        pass

    def refresh_table(self):
        """刷新表格"""
        self.load_table_data()

    def cv2_to_qpixmap(self, cv_img):
        """将OpenCV图像转换为QPixmap"""
        pass

    def closeEvent(self, event):
        """关闭事件处理"""
        if self.cap is not None:
            self.cap.release()
        if self.timer.isActive():
            self.timer.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())