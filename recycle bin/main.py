from PyQt5.QtSql import QSqlQuery, QSqlDatabase
from PyQt5.QtCore import QByteArray

import sys
import os
import cv2
import torch
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QFileDialog, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import QTimer, Qt, QByteArray
from PyQt5.QtGui import QImage, QPixmap
from Face_main.face_ui import Ui_Form
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from pathlib import Path

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


class MySqlite:
    def __init__(self):
        try:
            self.database = QSqlDatabase.addDatabase("QSQLITE")
            self.database.setDatabaseName("student.db")
            self.database.open()
            self.query = QSqlQuery()
        except Exception as e:
            print(f"无法建立连接，error:{e}")

    def operation_sql(self, sql, params=None):
        try:
            if params:
                if not self.query.prepare(sql):
                    return self.query.lastError().text()
                for param in params:
                    if isinstance(param, bytes):
                        self.query.addBindValue(QByteArray(param))
                    else:
                        self.query.addBindValue(param)
                if not self.query.exec_():
                    return self.query.lastError().text()
            else:
                if not self.query.exec_(sql):
                    return self.query.lastError().text()
            if sql.strip().upper().startswith("SELECT"):
                results = []
                while self.query.next():
                    row = []
                    for i in range(self.query.record().count()):
                        value = self.query.value(i)
                        if isinstance(value, QByteArray):
                            value = bytes(value)
                        row.append(value)
                    results.append(row)
                return results
            else:
                return True
        except Exception as e:
            return str(e)


class FaceNet:
    def __init__(self, yolo_model_path):
        self.facenet_model = InceptionResnetV1(pretrained='casia-webface').eval().to('cpu')
        self.yolo_model = YOLO(yolo_model_path)

    def preprocess_face_img(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (160, 160))
        face_tensor = (torch.tensor(face_img).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
        return face_tensor

    def facenet(self, face_img):
        face_tensor = self.preprocess_face_img(face_img=face_img)
        with torch.no_grad():
            face_embedding = self.facenet_model(face_tensor)
            l2_norm = torch.norm(face_embedding, p=2, dim=1, keepdim=True)
            face_embedding_normalized = face_embedding.div(l2_norm)
        return face_embedding_normalized.cpu().numpy()

    def getfacepos(self, img):
        face_imgs = []
        if isinstance(img, str):
            img_path = Path(img)
            if img_path.is_dir():
                for image_path in img_path.glob('*.jpg'):
                    frame = cv2.imread(str(image_path))
                    results = self.yolo_model.predict(frame, conf=0.7, verbose=False)
                    boxes = results[0].boxes
                    for box in boxes:
                        x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                        face_img = frame[y0:y1, x0:x1]
                        face_imgs.append(face_img)
            elif img_path.is_file():
                frame = cv2.imread(str(img_path))
                results = self.yolo_model.predict(frame, conf=0.7, verbose=False)
                boxes = results[0].boxes
                for box in boxes:
                    x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                    face_img = frame[y0:y1, x0:x1]
                    face_imgs.append(face_img)
        elif isinstance(img, np.ndarray):
            frame = img
            results = self.yolo_model.predict(frame, conf=0.7, verbose=False)
            boxes = results[0].boxes
            for box in boxes:
                x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                face_img = frame[y0:y1, x0:x1]
                face_imgs.append(face_img)
        return face_imgs

    def display_original_image(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        return pixmap


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.db = MySqlite()
        self.cap = None
        self.timer = QTimer()
        self.current_image = None
        self.current_faces = []

        self.yolo_path = r'/my_yolo/runs/detect/train/weights/best.pt'
        self.face_net = FaceNet(self.yolo_path)

        self.init_signals()
        self.create_table_if_not_exists()
        self.load_table_data()

    def init_signals(self):
        self.ui.pushButton_sb_3.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(0))
        self.ui.pushButton_25.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(1))
        self.ui.pushButton_gl_3.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(2))
        self.ui.pushButton_lr_3.clicked.connect(self.close)

        self.ui.pushButton_14.clicked.connect(self.upload_image_for_enrollment)
        self.ui.pushButton_15.clicked.connect(self.start_camera_enrollment)
        self.ui.pushButton_16.clicked.connect(self.capture_face_enrollment)
        self.ui.pushButton_17.clicked.connect(self.stop_camera_enrollment)
        self.ui.pushButton_26.clicked.connect(self.save_face_info)

        self.ui.pushButton_19.clicked.connect(self.upload_image_for_recognition)
        self.ui.pushButton_20.clicked.connect(self.start_camera_recognition)
        self.ui.pushButton_21.clicked.connect(self.stop_camera_recognition)

        self.ui.pushButton_22.clicked.connect(self.search_database)
        self.ui.pushButton_23.clicked.connect(self.delete_record)
        self.ui.pushButton_24.clicked.connect(self.refresh_table)

        self.timer.timeout.connect(self.update_frame)

    def create_table_if_not_exists(self):
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

    def set_pixmap_to_label(self, label, img, width=None, height=None):
        if width is None:
            width = label.width()
        if height is None:
            height = label.height()
        pixmap = self.face_net.display_original_image(img)
        label.setPixmap(pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def get_input_values(self):
        name = self.ui.lineEdit_13.text().strip()
        age = self.ui.lineEdit_14.text().strip()
        student_id = self.ui.lineEdit_15.text().strip()
        gender = "男" if self.ui.radioButton_5.isChecked() else ("女" if self.ui.radioButton_6.isChecked() else "")
        return name, age, student_id, gender

    def validate_inputs(self, name, age, student_id, gender):
        if not all([name, age, student_id, gender]):
            QMessageBox.warning(self, "警告", "请填写完整信息！")
            return False
        try:
            age = int(age)
        except ValueError:
            QMessageBox.warning(self, "警告", "年龄必须是数字！")
            return False
        return True

    def reset_enrollment_fields(self):
        self.ui.lineEdit_13.clear()
        self.ui.lineEdit_14.clear()
        self.ui.lineEdit_15.clear()
        self.ui.radioButton_5.setChecked(False)
        self.ui.radioButton_6.setChecked(False)
        self.ui.label_30.clear()
        self.current_faces = []

    def upload_image_for_enrollment(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.ui.lineEdit_7.setText(file_path)
            original_img = cv2.imread(file_path)
            if original_img is not None:
                self.set_pixmap_to_label(self.ui.label_15, original_img)
            face_imgs = self.face_net.getfacepos(file_path)
            if face_imgs:
                self.current_faces = face_imgs
                self.set_pixmap_to_label(self.ui.label_30, face_imgs[0])
                QMessageBox.information(self, "成功", f"检测到 {len(face_imgs)} 张人脸\n已自动选择第一张人脸")
            else:
                QMessageBox.warning(self, "警告", "未检测到人脸，请更换图片！")

    def start_camera_enrollment(self):
        if self.cap is not None and self.cap.isOpened():
            QMessageBox.warning(self, "警告", "摄像头已在运行中！")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头！请检查设备连接。")
            return
        self.timer.start(30)
        self.current_faces = []

    def capture_face_enrollment(self):
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "警告", "请先开启摄像头！")
            return
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "当前没有可用的摄像头画面！")
            return
        face_imgs = self.face_net.getfacepos(self.current_image)
        if face_imgs:
            self.current_faces = face_imgs
            self.set_pixmap_to_label(self.ui.label_30, face_imgs[0])
            QMessageBox.information(self, "成功", f"检测到 {len(face_imgs)} 张人脸\n已自动选择第一张人脸")
        else:
            QMessageBox.warning(self, "警告", "当前画面未检测到人脸！")

    def stop_camera_enrollment(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.ui.label_15.clear()
        self.ui.label_30.clear()

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_image = frame.copy()
                detected_frame = self.draw_face_boxes(frame)
                self.set_pixmap_to_label(self.ui.label_15, detected_frame)

    def draw_face_boxes(self, frame):
        processed_frame = frame.copy()
        results = self.face_net.yolo_model.predict(processed_frame, conf=0.7, verbose=False)
        boxes = results[0].boxes
        for box in boxes:
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(processed_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return processed_frame

    def save_face_info(self):
        name, age, student_id, gender = self.get_input_values()
        if not self.validate_inputs(name, age, student_id, gender):
            return
        if not self.current_faces:
            QMessageBox.warning(self, "警告", "没有检测到人脸！")
            return
        try:
            face_img = self.current_faces[0]
            embedding = self.face_net.facenet(face_img)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"人脸特征提取失败：{str(e)}")
            return
        _, img_encoded = cv2.imencode('.jpg', face_img)
        photo_bytes = img_encoded.tobytes()
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = "INSERT INTO student_info (姓名,年龄,性别,学号,录入时间,照片) VALUES (?, ?, ?, ?, ?, ?)"
        result = self.db.operation_sql(sql, [name, int(age), gender, student_id, current_time, photo_bytes])
        if result is True:
            QMessageBox.information(self, "成功", "人脸信息保存成功！")
            self.reset_enrollment_fields()
        else:
            QMessageBox.critical(self, "错误", f"保存失败：{result}")

    def upload_image_for_recognition(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择识别图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.ui.lineEdit_11.setText(file_path)
            original_img = cv2.imread(file_path)
            if original_img is not None:
                self.set_pixmap_to_label(self.ui.label_24, original_img)
                face_imgs = self.face_net.getfacepos(file_path)
                if face_imgs:
                    recognition_results = []
                    for i, face_img in enumerate(face_imgs):
                        try:
                            embedding = self.face_net.facenet(face_img)
                            match_result = self.find_matches(embedding)
                            if match_result:
                                name, confidence = match_result
                                recognition_results.append(f"人脸{i+1}: {name} (置信度: {confidence:.2f})")
                            else:
                                recognition_results.append(f"人脸{i+1}: 未知人员 (置信度: 0.00)")
                        except Exception as e:
                            recognition_results.append(f"人脸{i+1}: 特征提取失败 ({str(e)})")
                    result_text = "\n".join(recognition_results)
                    self.ui.plainTextEdit_2.setPlainText(result_text)
                else:
                    QMessageBox.warning(self, "警告", "未检测到人脸！")
                    self.ui.plainTextEdit_2.setPlainText("未检测到人脸！")
            else:
                QMessageBox.critical(self, "错误", "无法读取图片文件！")
                self.ui.plainTextEdit_2.setPlainText("无法读取图片文件！")

    def find_matches(self, query_embedding, threshold=1.0):
        sql = "SELECT ID, 姓名, 年龄, 性别, 学号, 录入时间, 照片 FROM student_info"
        result = self.db.operation_sql(sql)
        if not result or result is True:
            return None

        min_distance = float('inf')
        best_match = None
        for row in result:
            blob_data = row[6]
            if blob_data:
                nparr = np.frombuffer(blob_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    try:
                        db_embedding = self.face_net.facenet(img)
                        distance = np.linalg.norm(query_embedding - db_embedding)
                        if distance < min_distance and distance < threshold:
                            min_distance = distance
                            person_info = {'姓名': row[1], '年龄': row[2], '性别': row[3], '学号': row[4], '录入时间': row[5]}
                            best_match = (person_info, 1 - distance)
                    except Exception as e:
                        print(f"特征提取错误: {str(e)}")
                        continue
        return best_match if best_match else None

    def start_camera_recognition(self):
        if self.cap is not None and self.cap.isOpened():
            QMessageBox.warning(self, "警告", "摄像头已在运行中！")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头！请检查设备连接。")
            return

        self.recognition_locked = False
        self.locked_result = ""

        self.timer.timeout.disconnect()
        self.timer.timeout.connect(self.recognize_from_camera)
        self.timer.start(30)

    def recognize_from_camera(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if not self.recognition_locked:
                    result_frame, recognition_result = self.recognize_face_in_frame(frame)

                    if recognition_result and "没有该人员信息" not in recognition_result and recognition_result.strip() != "":
                        self.recognition_locked = True
                        self.locked_result = recognition_result
                        self.ui.plainTextEdit_2.setPlainText(recognition_result)
                    else:
                        self.set_pixmap_to_label(self.ui.label_24, result_frame)
                        self.ui.plainTextEdit_2.setPlainText(recognition_result)
                else:
                    result_frame, _ = self.recognize_face_in_frame(frame)
                    self.set_pixmap_to_label(self.ui.label_24, result_frame)
                    self.ui.plainTextEdit_2.setPlainText(self.locked_result)

    def recognize_face_in_frame(self, frame):
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

    def stop_camera_recognition(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.ui.label_24.clear()
        self.ui.plainTextEdit_2.clear()

    def search_database(self):
        keyword = self.ui.lineEdit_12.text().strip()
        if not keyword:
            QMessageBox.warning(self, "警告", "请输入查询关键词！")
            return

        sql = "SELECT ID, 姓名, 年龄, 性别, 学号, 录入时间 FROM student_info WHERE 姓名 LIKE ? OR 学号 LIKE ?"
        param = f"%{keyword}%"
        result = self.db.operation_sql(sql, [param, param])

        results = []
        if result and result is not True:
            results = result

        self.load_table_data(results)

    def load_table_data(self, data=None):
        self.ui.tableView_2.setRowCount(0)

        headers = ["ID", "姓名", "年龄", "性别", "学号", "录入时间"]
        self.ui.tableView_2.setColumnCount(len(headers))
        self.ui.tableView_2.setHorizontalHeaderLabels(headers)

        header = self.ui.tableView_2.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        if data is None:
            sql = "SELECT ID, 姓名, 年龄, 性别, 学号, 录入时间 FROM student_info"
            result = self.db.operation_sql(sql)
            if result and result is not True:
                data = result
            else:
                data = []

        if data:
            self.ui.tableView_2.setRowCount(len(data))
            for row_idx, row_data in enumerate(data):
                for col_idx, data in enumerate(row_data):
                    if isinstance(data, bytes):
                        data = "BLOB数据"
                    item = QTableWidgetItem(str(data) if data is not None else "")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.ui.tableView_2.setItem(row_idx, col_idx, item)

    def delete_record(self):
        selected_rows = self.ui.tableView_2.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请先选择要删除的记录！")
            return

        reply = QMessageBox.question(self, '确认删除', '确定要删除选中的记录吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            for index in sorted(selected_rows, key=lambda x: x.row(), reverse=True):
                row = index.row()
                id_item = self.ui.tableView_2.item(row, 0)
                if id_item:
                    record_id = id_item.text()
                    sql = "DELETE FROM student_info WHERE ID = ?"
                    result = self.db.operation_sql(sql, [record_id])
                    if result is not True:
                        QMessageBox.critical(self, "错误", f"删除记录 ID={record_id} 失败")

            self.load_table_data()
            QMessageBox.information(self, "成功", "记录删除成功！")

    def refresh_table(self):
        self.load_table_data()

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        if self.timer.isActive():
            self.timer.stop()
        event.accept()


if __name__ == '__main__':
    db = MySqlite()

    ret = db.operation_sql("""
        create table IF NOT EXISTS student_info(
            ID integer primary key AUTOINCREMENT,
            姓名 text,
            年龄 int,
            性别 text,
            学号 text,
            录入时间 text,
            照片 BLOB
        )
    """)
    print(ret)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())