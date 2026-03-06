# main_window.py
import sys
import os
import cv2
import torch
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QFileDialog, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import QTimer, Qt, QByteArray, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from face_ui import Ui_Form
from sqlite_db import MySqlite
from facenet_model import FaceNet

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


class CameraThread(QThread):
    # 定义信号，用于发送处理结果
    frame_processed = pyqtSignal(object, str)
    status_updated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.mode = None  # 'enrollment' or 'recognition'
        self.face_net = None
        self.cap = None
        self.face_cache = []
        self.threshold = 0.8

    def setup(self, cap, face_net, mode, face_cache, threshold):
        self.cap = cap
        self.face_net = face_net
        self.mode = mode
        self.face_cache = face_cache
        self.threshold = threshold

    def run(self):
        self.running = True
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    if self.mode == 'recognition':
                        processed_frame, recognition_result = self.recognize_face_in_frame(frame)
                        self.frame_processed.emit(processed_frame, recognition_result)
                    elif self.mode == 'enrollment':
                        processed_frame = self.draw_face_boxes(frame)
                        self.frame_processed.emit(processed_frame, "")
                else:
                    self.status_updated.emit("摄像头读取失败")
                    break
            else:
                self.status_updated.emit("摄像头未打开")
                break

    def stop(self):
        self.running = False

    def draw_face_boxes(self, frame):
        """在帧上绘制人脸检测框"""
        processed_frame = frame.copy()
        results = self.face_net.yolo_model.predict(processed_frame, conf=0.7, verbose=False)
        boxes = results[0].boxes
        for box in boxes:
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(processed_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return processed_frame

    def recognize_face_in_frame(self, frame):
        """识别人脸帧"""
        processed_frame = frame.copy()
        results = self.face_net.yolo_model.predict(processed_frame, conf=0.7, verbose=False)
        boxes = results[0].boxes
        recognition_result = ""

        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            face_img = processed_frame[y0:y1, x0:x1]

            try:
                embedding = self.face_net.facenet(face_img)
                match_result = self.find_matches(embedding)

                if match_result:
                    person_info, confidence = match_result
                    name = person_info['姓名']
                    age = person_info['年龄']
                    gender = person_info['性别']
                    student_id = person_info['学号']
                    record_time = person_info['录入时间']

                    recognition_result += f"人脸{i+1}:\n姓名: {name}\n年龄: {age}\n性别: {gender}\n学号: {student_id}\n录入时间: {record_time}\n置信度: {confidence:.2f}\n\n"
                else:
                    recognition_result += f"人脸{i+1}: 没有该人员信息\n\n"
            except Exception as e:
                recognition_result += f"人脸{i+1}: 特征提取失败 ({str(e)})\n\n"

            cv2.rectangle(processed_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        return processed_frame, recognition_result

    def find_matches(self, query_embedding):
        """查找数据库中最相似的人脸"""
        min_distance = float('inf')
        best_match = None

        for person_data in self.face_cache:
            db_embedding = person_data['embedding']
            distance = np.linalg.norm(query_embedding - db_embedding)

            if distance < min_distance and distance < self.threshold:
                min_distance = distance
                confidence = 1 - (distance / self.threshold)
                best_match = (person_data['info'], confidence)

        return best_match


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # 初始化数据库
        self.db = MySqlite()

        # 初始化摄像头相关变量
        self.cap = None
        self.camera_thread = None
        self.current_image = None
        self.current_faces = []

        self.yolo_path = r'C:\Users\Lenovo\Desktop\HQYJ\Facial_Recognition\my_yolo\runs\detect\train\weights\best.pt'
        self.face_net = FaceNet(self.yolo_path)

        # 初始化人脸缓存
        self.face_cache = []
        self._refresh_cache()

        # 绑定信号槽
        self.init_signals()

        # 初始化数据库表
        self.create_table_if_not_exists()

        self.load_table_data()

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
        if result:
            print("数据库表创建成功")
        else:
            print("数据库表创建失败")

    def _refresh_cache(self):
        """刷新人脸缓存"""
        self.face_cache = []
        sql = "SELECT ID, 姓名, 年龄, 性别, 学号, 录入时间, 照片 FROM student_info"
        result = self.db.operation_sql(sql)

        if result:
            for row in result:
                blob_data = row[6]
                if blob_data:
                    nparr = np.frombuffer(blob_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        try:
                            embedding = self.face_net.facenet(img)
                            person_info = {
                                'ID': row[0],
                                '姓名': row[1],
                                '年龄': row[2],
                                '性别': row[3],
                                '学号': row[4],
                                '录入时间': row[5]
                            }
                            self.face_cache.append({
                                'embedding': embedding,
                                'info': person_info
                            })
                        except Exception as e:
                            print(f"特征提取错误: {str(e)}")

    def upload_image_for_enrollment(self):
        """上传图片进行人脸录入"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            self.ui.lineEdit_7.setText(file_path)
            original_img = cv2.imread(file_path)
            if original_img is not None:
                pixmap1 = self.face_net.display_original_image(original_img)
                self.ui.label_15.setPixmap(pixmap1.scaled(
                    self.ui.label_15.width(), self.ui.label_15.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))

            face_imgs = self.face_net.getfacepos(file_path)
            if face_imgs:
                self.current_faces = face_imgs
                # 显示第一张检测到的人脸
                pixmap2 = self.face_net.display_original_image(face_imgs[0])
                self.ui.label_30.setPixmap(pixmap2.scaled(
                    self.ui.label_30.width(), self.ui.label_30.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))

                QMessageBox.information(
                    self,
                    "成功",
                    f"检测到 {len(face_imgs)} 张人脸\n已自动选择第一张人脸"
                )
            else:
                QMessageBox.warning(self, "警告", "未检测到人脸，请更换图片！")

    def start_camera_enrollment(self):
        """开启摄像头进行人脸录入"""
        if self.cap is not None and self.cap.isOpened():
            QMessageBox.warning(self, "警告", "摄像头已在运行中！")
            return

        # 尝试打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头！请检查设备连接。")
            return

        # 创建并启动摄像头线程
        self.camera_thread = CameraThread()
        self.camera_thread.setup(self.cap, self.face_net, 'enrollment', self.face_cache, 0.8)
        self.camera_thread.frame_processed.connect(self.update_frame)
        self.camera_thread.start()

    def capture_face_enrollment(self):
        """拍照进行人脸录入"""
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "警告", "请先开启摄像头！")
            return
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "当前没有可用的摄像头画面！")
            return

        # 在当前画面中检测人脸
        face_imgs = self.face_net.getfacepos(self.current_image)
        if face_imgs:
            self.current_faces = face_imgs
            # 显示检测到的第一张人脸
            pixmap1 = self.face_net.display_original_image(face_imgs[0])
            self.ui.label_30.setPixmap(pixmap1.scaled(
                self.ui.label_30.width(), self.ui.label_30.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            QMessageBox.information(
                self,
                "成功",
                f"检测到 {len(face_imgs)} 张人脸\n已自动选择第一张人脸"
            )
        else:
            QMessageBox.warning(self, "警告", "当前画面未检测到人脸！")

    def stop_camera_enrollment(self):
        """关闭摄像头"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # 清空显示
        self.ui.label_15.clear()
        self.ui.label_30.clear()

    def update_frame(self, frame, recognition_result):
        """更新摄像头帧"""
        if frame is not None:
            # 保存当前帧用于后续人脸捕获
            self.current_image = frame.copy()
            # 显示原始摄像头画面到中间label
            pixmap1 = self.face_net.display_original_image(frame)
            self.ui.label_15.setPixmap(pixmap1.scaled(
                self.ui.label_15.width(), self.ui.label_15.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

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
            embedding = self.face_net.facenet(face_img)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"人脸特征提取失败：{str(e)}")
            return
        # 将图像转换为字节流存储
        _, img_encoded = cv2.imencode('.jpg', face_img)
        photo_bytes = img_encoded.tobytes()
        # 插入数据库
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = """
                INSERT INTO student_info (姓名,年龄,性别,学号,录入时间,照片)
                VALUES (?, ?, ?, ?, ?, ?)
                """
        result = self.db.operation_sql(sql, [name, age, gender, student_id, current_time, photo_bytes])
        if result:
            QMessageBox.information(self, "成功", "人脸信息保存成功！")
            # 清空输入框和显示
            self.ui.lineEdit_13.clear()
            self.ui.lineEdit_14.clear()
            self.ui.lineEdit_15.clear()
            self.ui.radioButton_5.setChecked(False)
            self.ui.radioButton_6.setChecked(False)
            self.ui.label_30.clear()
            self.current_faces = []

            # 刷新缓存
            self._refresh_cache()
        else:
            QMessageBox.critical(self, "错误", "保存失败")

    def upload_image_for_recognition(self):
        """上传图片进行人脸识别"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择识别图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.ui.lineEdit_11.setText(file_path)
            original_img = cv2.imread(file_path)
            if original_img is not None:
                # 显示原始图像
                pixmap1 = self.face_net.display_original_image(original_img)
                self.ui.label_24.setPixmap(pixmap1.scaled(
                    self.ui.label_24.width(), self.ui.label_24.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                # 检测人脸
                face_imgs = self.face_net.getfacepos(file_path)
                if face_imgs:
                    recognition_results = []
                    for i, face_img in enumerate(face_imgs):
                        # 提取人脸特征
                        try:
                            embedding = self.face_net.facenet(face_img)
                            # 查找匹配的人脸
                            match_result = self.find_matches_from_cache(embedding)
                            if match_result:
                                name, confidence = match_result
                                recognition_results.append(f"人脸{i+1}: {name} (置信度: {confidence:.2f})")
                            else:
                                recognition_results.append(f"人脸{i+1}: 未知人员 (置信度: 0.00)")
                        except Exception as e:
                            recognition_results.append(f"人脸{i+1}: 特征提取失败 ({str(e)})")
                    # 更新识别结果显示
                    result_text = "\n".join(recognition_results)
                    self.ui.plainTextEdit_2.setPlainText(result_text)
                else:
                    QMessageBox.warning(self, "警告", "未检测到人脸！")
                    self.ui.plainTextEdit_2.setPlainText("未检测到人脸！")
            else:
                QMessageBox.critical(self, "错误", "无法读取图片文件！")
                self.ui.plainTextEdit_2.setPlainText("无法读取图片文件！")

    def find_matches_from_cache(self, query_embedding, threshold=0.8):
        """从缓存中查找最相似的人脸"""
        min_distance = float('inf')
        best_match = None

        for person_data in self.face_cache:
            db_embedding = person_data['embedding']
            distance = np.linalg.norm(query_embedding - db_embedding)

            if distance < min_distance and distance < threshold:
                min_distance = distance
                confidence = 1 - (distance / threshold)
                best_match = (person_data['info']['姓名'], confidence)

        return best_match

    def start_camera_recognition(self):
        """开启摄像头进行人脸识别"""
        if self.cap is not None and self.cap.isOpened():
            QMessageBox.warning(self, "警告", "摄像头已在运行中！")
            return

        # 尝试打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头！请检查设备连接。")
            return

        # 创建并启动摄像头线程
        self.camera_thread = CameraThread()
        self.camera_thread.setup(self.cap, self.face_net, 'recognition', self.face_cache, 0.8)
        self.camera_thread.frame_processed.connect(self.update_recognition_frame)
        self.camera_thread.start()

    def update_recognition_frame(self, frame, recognition_result):
        """更新识别帧"""
        if frame is not None:
            # 显示处理后的帧
            pixmap1 = self.face_net.display_original_image(frame)
            self.ui.label_24.setPixmap(pixmap1.scaled(
                self.ui.label_24.width(), self.ui.label_24.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            # 更新识别结果文本
            self.ui.plainTextEdit_2.setPlainText(recognition_result)

    def stop_camera_recognition(self):
        """关闭摄像头识别"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        # 清空显示
        self.ui.label_24.clear()
        self.ui.plainTextEdit_2.clear()

    def search_database(self):
        """搜索数据库"""
        # 简单实现：根据学号或姓名搜索
        keyword = self.ui.lineEdit_12.text().strip()
        if not keyword:
            QMessageBox.warning(self, "警告", "请输入查询关键词！")
            return

        # 模糊查询 姓名 或 学号
        sql = "SELECT ID, 姓名, 年龄, 性别, 学号, 录入时间 FROM student_info WHERE 姓名 LIKE ? OR 学号 LIKE ?"
        param = f"%{keyword}%"
        result = self.db.operation_sql(sql, [param, param])

        # 清空并重新填充表格
        self.ui.tableView_2.setRowCount(0)
        headers = ["ID", "姓名", "年龄", "性别", "学号", "录入时间"]
        self.ui.tableView_2.setColumnCount(len(headers))
        self.ui.tableView_2.setHorizontalHeaderLabels(headers)
        header = self.ui.tableView_2.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        if result:
            self.ui.tableView_2.setRowCount(len(result))
            for row_idx, row_data in enumerate(result):
                for col_idx, data in enumerate(row_data):
                    if isinstance(data, bytes):
                        data = "BLOB数据"
                    item = QTableWidgetItem(str(data) if data is not None else "")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.ui.tableView_2.setItem(row_idx, col_idx, item)

    def load_table_data(self):
        """加载表格数据"""
        # 清空现有表格
        self.ui.tableView_2.setRowCount(0)

        # 设置表头
        headers = ["ID", "姓名", "年龄", "性别", "学号", "录入时间"]
        self.ui.tableView_2.setColumnCount(len(headers))
        self.ui.tableView_2.setHorizontalHeaderLabels(headers)

        # 设置列宽自适应
        header = self.ui.tableView_2.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        # 查询数据库
        sql = "SELECT ID, 姓名, 年龄, 性别, 学号, 录入时间 FROM student_info"
        result = self.db.operation_sql(sql)

        if result:
            self.ui.tableView_2.setRowCount(len(result))
            for row_idx, row_data in enumerate(result):
                for col_idx, data in enumerate(row_data):
                    # 处理可能存在的字节数据或其他类型，确保转为字符串
                    if isinstance(data, bytes):
                        data = "BLOB数据" # 照片中不直接显示二进制
                    item = QTableWidgetItem(str(data) if data is not None else "")
                    item.setTextAlignment(Qt.AlignCenter) # 居中对齐
                    self.ui.tableView_2.setItem(row_idx, col_idx, item)

    def delete_record(self):
        """删除记录"""
        # 获取当前选中的行
        selected_rows = self.ui.tableView_2.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请先选择要删除的记录！")
            return

        # 确认删除
        reply = QMessageBox.question(self, '确认删除', '确定要删除选中的记录吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 从后往前删除，避免索引变化
            for index in sorted(selected_rows, key=lambda x: x.row(), reverse=True):
                row = index.row()
                # 获取该行的 ID (假设第一列是 ID)
                id_item = self.ui.tableView_2.item(row, 0)
                if id_item:
                    record_id = id_item.text()
                    sql = "DELETE FROM student_info WHERE ID = ?"
                    result = self.db.operation_sql(sql, [record_id])
                    if not result:
                        QMessageBox.critical(self, "错误", f"删除记录 ID={record_id} 失败")

            # 删除后刷新表格
            self.load_table_data()
            # 刷新缓存
            self._refresh_cache()
            QMessageBox.information(self, "成功", "记录删除成功！")

    def refresh_table(self):
        """刷新表格"""
        self.load_table_data()

    def closeEvent(self, event):
        """关闭事件处理"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()

        if self.cap is not None:
            self.cap.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())