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
from PyQT.sq_db import MySqlite
from my_facenet.facenet_model2 import getfacepos, facenet

# 移除可能冲突的环境变量
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
            # 读取图片并显示在左侧Label
            pixmap = QPixmap(file_path)
            self.ui.label_15.setPixmap(pixmap.scaled(self.ui.label_15.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # 检测人脸并显示在右侧Label
            face_img, face_pos = self.detect_face_in_image(file_path)
            if face_img is not None:
                self.current_image = face_img
                face_pixmap = self.cv2_to_qpixmap(face_img)
                self.ui.label_30.setPixmap(face_pixmap.scaled(self.ui.label_30.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                QMessageBox.warning(self, "警告", "未检测到人脸")

    def detect_face_in_image(self, image_path):
        """检测图片中的人脸"""
        try:
            # 使用YOLO检测人脸位置
            face_imgs, positions = getfacepos(r"C:\Users\Lenovo\Desktop\HQYJ\Facial_Recognition\my_yolo\runs\detect\train\weights\best.pt", image_path)
            if len(face_imgs) > 0:
                return face_imgs[0], positions[0]  # 返回第一张人脸
            else:
                return None, None
        except Exception as e:
            print(f"人脸检测出错: {e}")
            return None, None

    def start_camera_enrollment(self):
        """开启摄像头进行人脸录入"""
        if self.cap is not None:
            self.stop_camera_enrollment()
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头")
            return
        
        self.timer.start(30)  # 每30ms更新一次画面

    def capture_face_enrollment(self):
        """拍照进行人脸录入"""
        if self.current_image is not None:
            face_pixmap = self.cv2_to_qpixmap(self.current_image)
            self.ui.label_30.setPixmap(face_pixmap.scaled(self.ui.label_30.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            QMessageBox.information(self, "提示", "人脸已捕获，请填写信息并保存")
        else:
            QMessageBox.warning(self, "警告", "请先开启摄像头并检测到人脸")

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
                # 显示原始画面
                original_pixmap = self.cv2_to_qpixmap(frame)
                self.ui.label_15.setPixmap(original_pixmap.scaled(self.ui.label_15.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

                # 检测人脸并绘制边界框
                processed_frame, faces = self.detect_and_draw_faces(frame)
                
                # 如果检测到人脸，保存最新的人脸图像
                if len(faces) > 0:
                    # 使用最后一张检测到的人脸
                    self.current_image = faces[-1]
                
                # 显示处理后的画面
                processed_pixmap = self.cv2_to_qpixmap(processed_frame)
                self.ui.label_15.setPixmap(processed_pixmap.scaled(self.ui.label_15.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def detect_and_draw_faces(self, frame):
        """检测人脸并绘制边界框"""
        try:
            # 使用YOLO检测人脸
            face_imgs, positions = getfacepos(r'C:\Users\Lenovo\Desktop\HQYJ\Facial_Recognition\my_yolo\runs\detect\train\weights\best.pt', frame)
            
            # 在原图上绘制边界框
            output_frame = frame.copy()
            for pos in positions:
                x0, y0, x1, y1 = pos
                cv2.rectangle(output_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            
            return output_frame, face_imgs
        except Exception as e:
            print(f"人脸检测出错: {e}")
            return frame, []

    def save_face_info(self):
        """保存人脸信息到数据库"""
        name = self.ui.lineEdit_13.text()
        age = self.ui.lineEdit_14.text()
        student_id = self.ui.lineEdit_15.text()
        
        # 获取性别
        gender = ""
        if self.ui.radioButton_5.isChecked():
            gender = "男"
        elif self.ui.radioButton_6.isChecked():
            gender = "女"
        
        # 检查必要信息是否填写
        if not name or not age or not student_id or not gender or self.current_image is None:
            QMessageBox.warning(self, "警告", "请填写完整信息并确保有人脸图像")
            return
        
        # 将图片转换为字节
        _, img_encoded = cv2.imencode('.jpg', self.current_image)
        img_bytes = img_encoded.tobytes()
        
        # 插入数据库
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = "INSERT INTO student_info (姓名, 年龄, 性别, 学号, 录入时间, 照片) VALUES (?, ?, ?, ?, ?, ?)"
        result = self.db.operation_sql(sql, [name, int(age), gender, student_id, current_time, img_bytes])
        
        if result is True:
            QMessageBox.information(self, "成功", "人脸信息保存成功")
            # 清空输入框
            self.ui.lineEdit_13.clear()
            self.ui.lineEdit_14.clear()
            self.ui.lineEdit_15.clear()
            self.ui.radioButton_5.setChecked(False)
            self.ui.radioButton_6.setChecked(False)
            self.ui.label_30.clear()
            self.current_image = None
        else:
            QMessageBox.critical(self, "错误", f"保存失败: {result}")

    def upload_image_for_recognition(self):
        """上传图片进行人脸识别"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # 读取图片并显示
            pixmap = QPixmap(file_path)
            self.ui.label_24.setPixmap(pixmap.scaled(self.ui.label_24.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # 识别人脸
            result_text = self.recognize_face_in_image(file_path)
            self.ui.plainTextEdit_2.setPlainText(result_text)

    def recognize_face_in_image(self, image_path):
        """识别人脸图片"""
        try:
            # 检测人脸
            face_imgs, positions = getfacepos(r'C:\Users\Lenovo\Desktop\HQYJ\Facial_Recognition\my_yolo\runs\detect\train\weights\best.pt', image_path)
            
            if len(face_imgs) == 0:
                return "未检测到人脸"
            
            results = []
            for i, face_img in enumerate(face_imgs):
                # 提取特征向量
                embedding = facenet(face_img)
                
                # 查询数据库匹配
                matches = self.find_matches(embedding)
                
                if matches:
                    result = f"第{i+1}个人脸: 匹配到 {matches[0][0]}, 置信度: {matches[0][1]:.2f}"
                else:
                    result = f"第{i+1}个人脸: 未找到匹配项"
                
                results.append(result)
            
            return "\n".join(results)
        except Exception as e:
            return f"人脸识别出错: {e}"

    def find_matches(self, query_embedding, threshold=0.6):
        """查找数据库中最相似的人脸"""
        try:
            # 查询所有已知人脸
            sql = "SELECT ID, 姓名, 照片 FROM student_info"
            result = self.db.operation_sql(sql)
            
            if result is not True:
                return []
            
            matches = []
            self.db.query.first()
            record = self.db.query.record()
            
            while self.db.query.isValid():
                photo_data = self.db.query.value(2)
                if photo_data:
                    # 将BLOB数据转换为numpy数组
                    if isinstance(photo_data, QByteArray):
                        photo_data = photo_data.data()
                    
                    # 解码图片
                    nparr = np.frombuffer(photo_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        # 提取特征向量
                        known_embedding = facenet(img)
                        
                        # 计算相似度
                        distance = np.linalg.norm(query_embedding - known_embedding)
                        
                        if distance < threshold:
                            name = self.db.query.value(1)
                            matches.append((name, 1 - distance/threshold))
                
                self.db.query.next()
            
            # 按相似度排序
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[:5]  # 返回前5个最相似的结果
            
        except Exception as e:
            print(f"查找匹配出错: {e}")
            return []

    def start_camera_recognition(self):
        """开启摄像头进行人脸识别"""
        if self.cap is not None:
            self.stop_camera_recognition()
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头")
            return
        
        # 启动人脸识别专用定时器
        self.timer.timeout.disconnect()
        self.timer.timeout.connect(self.recognize_from_camera)
        self.timer.start(100)  # 每100ms处理一次

    def recognize_from_camera(self):
        """从摄像头进行人脸识别"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 显示原始画面
                original_pixmap = self.cv2_to_qpixmap(frame)
                self.ui.label_24.setPixmap(original_pixmap.scaled(self.ui.label_24.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
                # 识别人脸
                result_text = self.recognize_face_in_frame(frame)
                self.ui.plainTextEdit_2.setPlainText(result_text)

    def recognize_face_in_frame(self, frame):
        """识别人脸帧"""
        try:
            # 检测人脸
            face_imgs, positions = getfacepos(r'C:\Users\Lenovo\Desktop\HQYJ\Facial_Recognition\my_yolo\runs\detect\train\weights\best.pt', frame)
            
            if len(face_imgs) == 0:
                return "未检测到人脸"
            
            results = []
            for i, face_img in enumerate(face_imgs):
                # 提取特征向量
                embedding = facenet(face_img)
                
                # 查询数据库匹配
                matches = self.find_matches(embedding)
                
                if matches:
                    result = f"第{i+1}个人脸: {matches[0][0]}"
                else:
                    result = f"第{i+1}个人脸: 未知"
                
                results.append(result)
            
            return "\n".join(results)
        except Exception as e:
            return f"人脸识别出错: {e}"

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
        search_term = self.ui.lineEdit_12.text()
        if search_term:
            sql = f"SELECT * FROM student_info WHERE 姓名 LIKE '%{search_term}%' OR 学号 LIKE '%{search_term}%'"
        else:
            sql = "SELECT * FROM student_info"
        
        result = self.db.operation_sql(sql)
        if result is True:
            self.load_table_data()
        else:
            QMessageBox.critical(self, "错误", f"查询失败: {result}")

    def load_table_data(self):
        """加载表格数据"""
        sql = "SELECT * FROM student_info"
        result = self.db.operation_sql(sql)
        
        if result is not True:
            return
        
        # 获取列数
        self.db.query.first()
        record = self.db.query.record()
        column_count = record.count()
        
        # 设置表格列数和列标题
        self.ui.tableView_2.setColumnCount(column_count)
        headers = ["ID", "姓名", "年龄", "性别", "学号", "录入时间", "照片"]
        self.ui.tableView_2.setHorizontalHeaderLabels(headers)
        
        # 计算行数
        row_count = 0
        self.db.query.first()
        while self.db.query.isValid():
            row_count += 1
            self.db.query.next()
        
        # 设置行数
        self.ui.tableView_2.setRowCount(row_count)
        
        # 填充数据
        self.db.query.first()
        row = 0
        while self.db.query.isValid():
            for col in range(column_count):
                value = self.db.query.value(col)
                # 如果是照片列，显示缩略图或提示
                if col == 6:  # 照片列
                    item_value = "[照片]" if value else ""
                else:
                    item_value = str(value) if value is not None else ""
                self.ui.tableView_2.setItem(row, col, QTableWidgetItem(item_value))
            row += 1
            self.db.query.next()

    def delete_record(self):
        """删除记录"""
        current_row = self.ui.tableView_2.currentRow()
        if current_row >= 0:
            item = self.ui.tableView_2.item(current_row, 0)  # ID列
            if item:
                student_id = item.text()
                reply = QMessageBox.question(self, "确认", f"确定要删除ID为 {student_id} 的记录吗？", 
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    sql = f"DELETE FROM student_info WHERE ID = {student_id}"
                    result = self.db.operation_sql(sql)
                    if result is True:
                        QMessageBox.information(self, "成功", "记录删除成功")
                        self.refresh_table()
                    else:
                        QMessageBox.critical(self, "错误", f"删除失败: {result}")
        else:
            QMessageBox.warning(self, "警告", "请选择要删除的记录")

    def refresh_table(self):
        """刷新表格"""
        self.load_table_data()

    def cv2_to_qpixmap(self, cv_img):
        """将OpenCV图像转换为QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

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