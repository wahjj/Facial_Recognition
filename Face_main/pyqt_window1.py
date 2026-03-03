import sys
import os
import cv2
import torch
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QFileDialog, QTableWidgetItem
from PyQt5.QtCore import QTimer, Qt, QByteArray
from PyQt5.QtGui import QImage, QPixmap
from face_ui import Ui_Form
from PyQT.sq_db import MySqlite
from my_facenet.facenet_model2 import getfacepos, facenet

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

        # 初始化模型路径
        self.yolo_weights_path = r'C:\Users\Lenovo\Desktop\HQYJ\Facial_Recognition\my_yolo\runs\detect\train\weights\best.pt'

        # 绑定信号槽
        self.init_signals()

        # 初始化数据库表
        self.create_table_if_not_exists()

        # 初始化表格
        self.init_table()

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

    def init_table(self):
        """初始化表格"""
        headers = ["ID", "姓名", "年龄", "性别", "学号", "录入时间"]
        self.ui.tableView_2.setColumnCount(len(headers))
        self.ui.tableView_2.setHorizontalHeaderLabels(headers)
        self.ui.tableView_2.setEditTriggers(self.ui.tableView_2.NoEditTriggers)

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
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.ui.lineEdit_7.setText(file_path)
            self.detect_face_in_image(file_path)

    def detect_face_in_image(self, image_path):
        """检测图片中的人脸"""
        try:
            face_imgs, positions = getfacepos(self.yolo_weights_path, image_path)
            if face_imgs:
                self.current_faces = face_imgs
                # 显示第一张检测到的人脸
                self.display_face_image(face_imgs[0])
            else:
                QMessageBox.warning(self, "警告", "未检测到人脸！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"人脸检测失败: {str(e)}")

    def display_face_image(self, face_img):
        """显示人脸图像到右侧预览框"""
        h, w, ch = face_img.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(face_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.rgbSwapped()
        self.ui.label_30.setPixmap(QPixmap.fromImage(p).scaled(
            self.ui.label_30.width(), self.ui.label_30.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_camera_enrollment(self):
        """开启摄像头进行人脸录入"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头！")
            return

        self.timer.start(30)  # 每30ms更新一次

    def capture_face_enrollment(self):
        """拍照进行人脸录入"""
        if self.current_image is not None:
            try:
                face_imgs, positions = getfacepos(self.yolo_weights_path, self.current_image)
                if face_imgs:
                    self.current_faces = face_imgs
                    # 显示第一张检测到的人脸
                    self.display_face_image(face_imgs[0])
                else:
                    QMessageBox.warning(self, "警告", "未检测到人脸！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"人脸检测失败: {str(e)}")
        else:
            QMessageBox.warning(self, "警告", "没有可用的图像！")

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
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_image = frame.copy()
                # 检测并绘制人脸
                processed_frame = self.detect_and_draw_faces(frame)
                # 转换为QPixmap并显示
                pixmap = self.cv2_to_qpixmap(processed_frame)
                self.ui.label_15.setPixmap(pixmap.scaled(
                    self.ui.label_15.width(), self.ui.label_15.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def detect_and_draw_faces(self, frame):
        """检测人脸并绘制边界框"""
        try:
            face_imgs, positions = getfacepos(self.yolo_weights_path, frame)
            for pos in positions:
                x0, y0, x1, y1 = pos
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            return frame
        except:
            return frame

    def save_face_info(self):
        """保存人脸信息到数据库"""
        # 获取输入信息
        name = self.ui.lineEdit_13.text().strip()
        age = self.ui.lineEdit_14.text().strip()
        student_id = self.ui.lineEdit_15.text().strip()

        # 检查性别选择
        gender = ""
        if self.ui.radioButton_5.isChecked():  # 男
            gender = "男"
        elif self.ui.radioButton_6.isChecked():  # 女
            gender = "女"

        # 验证输入
        if not name or not age or not student_id or not gender:
            QMessageBox.warning(self, "警告", "请填写完整信息！")
            return

        try:
            age = int(age)
        except ValueError:
            QMessageBox.warning(self, "警告", "年龄必须是数字！")
            return

        if not self.current_faces:
            QMessageBox.warning(self, "警告", "没有检测到人脸！")
            return

        # 计算人脸特征向量
        try:
            face_img = self.current_faces[0]
            embedding = facenet(face_img)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"人脸特征提取失败: {str(e)}")
            return

        # 将图像转换为字节流存储
        _, img_encoded = cv2.imencode('.jpg', face_img)
        photo_bytes = img_encoded.tobytes()

        # 插入数据库
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = """
        INSERT INTO student_info (姓名, 年龄, 性别, 学号, 录入时间, 照片)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        result = self.db.operation_sql(sql, [name, age, gender, student_id, current_time, photo_bytes])

        if result is True:
            QMessageBox.information(self, "成功", "人脸信息保存成功！")
            # 清空输入框
            self.ui.lineEdit_13.clear()
            self.ui.lineEdit_14.clear()
            self.ui.lineEdit_15.clear()
            self.ui.radioButton_5.setChecked(False)
            self.ui.radioButton_6.setChecked(False)
            self.ui.label_30.clear()
        else:
            QMessageBox.critical(self, "错误", f"保存失败: {result}")

    def upload_image_for_recognition(self):
        """上传图片进行人脸识别"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.ui.lineEdit_11.setText(file_path)
            self.recognize_face_in_image(file_path)

    def recognize_face_in_image(self, image_path):
        """识别人脸图片"""
        try:
            face_imgs, positions = getfacepos(self.yolo_weights_path, image_path)
            if not face_imgs:
                self.ui.plainTextEdit_2.setPlainText("未检测到人脸！")
                return

            results_text = []
            for i, face_img in enumerate(face_imgs):
                embedding = facenet(face_img)
                match_result = self.find_matches(embedding)

                if match_result:
                    results_text.append(f"人脸 {i+1}: 匹配成功 - {match_result}")
                else:
                    results_text.append(f"人脸 {i+1}: 未匹配到已知人脸")

            self.ui.plainTextEdit_2.setPlainText("\n".join(results_text))

            # 显示原图
            original_img = cv2.imread(image_path)
            processed_img = original_img.copy()
            for pos in positions:
                x0, y0, x1, y1 = pos
                cv2.rectangle(processed_img, (x0, y0), (x1, y1), (0, 255, 0), 2)

            pixmap = self.cv2_to_qpixmap(processed_img)
            self.ui.label_24.setPixmap(pixmap.scaled(
                self.ui.label_24.width(), self.ui.label_24.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))

        except Exception as e:
            QMessageBox.critical(self, "错误", f"人脸识别失败: {str(e)}")

    def find_matches(self, query_embedding, threshold=0.6):
        """查找数据库中最相似的人脸"""
        # 获取数据库中所有人脸特征
        sql = "SELECT ID, 姓名, 年龄, 性别, 学号, 照片 FROM student_info"
        self.db.query.exec_(sql)

        min_distance = float('inf')
        best_match = None

        while self.db.query.next():
            # 获取照片数据
            blob_data = self.db.query.value(5)
            if hasattr(blob_data, 'data'):
                blob_data = blob_data.data()

            if blob_data:
                # 解码图像
                nparr = np.frombuffer(blob_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    try:
                        # 提取特征
                        db_embedding = facenet(img)

                        # 计算欧氏距离
                        distance = np.linalg.norm(query_embedding - db_embedding)

                        if distance < min_distance and distance < threshold:
                            min_distance = distance
                            best_match = {
                                '姓名': self.db.query.value(1),
                                '年龄': self.db.query.value(2),
                                '性别': self.db.query.value(3),
                                '学号': self.db.query.value(4)
                            }
                    except:
                        continue

        if best_match:
            return f"{best_match['姓名']} ({best_match['学号']}, {best_match['年龄']}岁, {best_match['性别']})"
        else:
            return None

    def start_camera_recognition(self):
        """开启摄像头进行人脸识别"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头！")
            return

        self.timer.stop()  # 先停止其他计时器
        self.timer.timeout.disconnect()  # 断开之前的连接
        self.timer.timeout.connect(self.recognize_from_camera)
        self.timer.start(30)  # 每30ms更新一次

    def recognize_from_camera(self):
        """从摄像头进行人脸识别"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 检测并识别
                processed_frame = self.recognize_face_in_frame(frame)
                # 转换为QPixmap并显示
                pixmap = self.cv2_to_qpixmap(processed_frame)
                self.ui.label_24.setPixmap(pixmap.scaled(
                    self.ui.label_24.width(), self.ui.label_24.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def recognize_face_in_frame(self, frame):
        """识别人脸帧"""
        try:
            face_imgs, positions = getfacepos(self.yolo_weights_path, frame)
            results_text = []

            for i, (face_img, pos) in enumerate(zip(face_imgs, positions)):
                x0, y0, x1, y1 = pos
                embedding = facenet(face_img)
                match_result = self.find_matches(embedding)

                if match_result:
                    results_text.append(f"人脸 {i+1}: {match_result}")
                    # 标记匹配成功
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(frame, match_result.split('(')[0],
                                (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                else:
                    results_text.append(f"人脸 {i+1}: 未知人脸")
                    # 标记未匹配
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown",
                                (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)

            self.ui.plainTextEdit_2.setPlainText("\n".join(results_text))
            return frame
        except:
            return frame

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
        search_text = self.ui.lineEdit_12.text().strip()
        if search_text:
            sql = f"SELECT * FROM student_info WHERE 姓名 LIKE '%{search_text}%' OR 学号 LIKE '%{search_text}%'"
        else:
            sql = "SELECT * FROM student_info"

        self.db.query.exec_(sql)

        # 清空表格
        self.ui.tableView_2.setRowCount(0)

        row = 0
        while self.db.query.next():
            self.ui.tableView_2.insertRow(row)
            for col in range(6):  # ID, 姓名, 年龄, 性别, 学号, 录入时间
                item = QTableWidgetItem(str(self.db.query.value(col)))
                self.ui.tableView_2.setItem(row, col, item)
            row += 1

    def load_table_data(self):
        """加载表格数据"""
        sql = "SELECT * FROM student_info"
        self.db.query.exec_(sql)

        # 清空表格
        self.ui.tableView_2.setRowCount(0)

        row = 0
        while self.db.query.next():
            self.ui.tableView_2.insertRow(row)
            for col in range(6):  # ID, 姓名, 年龄, 性别, 学号, 录入时间
                item = QTableWidgetItem(str(self.db.query.value(col)))
                self.ui.tableView_2.setItem(row, col, item)
            row += 1

    def delete_record(self):
        """删除记录"""
        selected_rows = []
        for index in self.ui.tableView_2.selectionModel().selectedRows():
            selected_rows.append(index.row())

        if not selected_rows:
            QMessageBox.warning(self, "警告", "请选择要删除的记录！")
            return

        # 获取要删除的ID
        ids_to_delete = []
        for row in selected_rows:
            id_item = self.ui.tableView_2.item(row, 0)  # ID列
            if id_item:
                ids_to_delete.append(id_item.text())

        # 确认删除
        reply = QMessageBox.question(self, "确认删除",
                                     f"确定要删除选中的 {len(ids_to_delete)} 条记录吗？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            for id_val in ids_to_delete:
                sql = f"DELETE FROM student_info WHERE ID = {id_val}"
                result = self.db.operation_sql(sql)
                if result is not True:
                    QMessageBox.critical(self, "错误", f"删除ID为 {id_val} 的记录失败: {result}")

            # 刷新表格
            self.refresh_table()

    def refresh_table(self):
        """刷新表格"""
        self.load_table_data()

    def cv2_to_qpixmap(self, cv_img):
        """将OpenCV图像转换为QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_qt_format)

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