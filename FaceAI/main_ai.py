# main.py
import sys
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QFileDialog, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

# 导入自定义模块
from db import MySqlite
from facenet import FaceNet
from face_ui import Ui_Form

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


# ---------- 摄像头线程 ----------
class CameraThread(QThread):
    frame_processed = pyqtSignal(object, str)  # 发送处理后的图像和识别文本（或空字符串）
    status_signal = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None
        self.mode = "enroll"  # "enroll" 或 "recognize"
        self.face_net = None   # 由主窗口设置
        self.db_cache = None   # 特征缓存列表，由主窗口设置

    def set_face_net(self, face_net):
        self.face_net = face_net

    def set_db_cache(self, db_cache):
        self.db_cache = db_cache

    def set_mode(self, mode):
        self.mode = mode

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.status_signal.emit("无法打开摄像头")
            return

        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.status_signal.emit("摄像头读取失败")
                break

            # 根据模式处理帧
            if self.mode == "enroll":
                processed_frame, _ = self._process_enroll_frame(frame)
                self.frame_processed.emit(processed_frame, "")
            else:  # recognize
                processed_frame, result_text = self._process_recognize_frame(frame)
                self.frame_processed.emit(processed_frame, result_text)

        self.cap.release()

    def _process_enroll_frame(self, frame):
        boxes, _ = self.face_net.detect_faces(frame)
        frame_with_boxes = self.face_net.draw_boxes(frame.copy(), boxes)
        return frame_with_boxes, None

    def _process_recognize_frame(self, frame):
        boxes, face_imgs = self.face_net.detect_faces(frame)
        frame_with_boxes = self.face_net.draw_boxes(frame.copy(), boxes)
        result_lines = []
        for i, face_img in enumerate(face_imgs):
            try:
                query_emb = self.face_net.get_embedding(face_img)
                match = self._find_best_match(query_emb)
                if match:
                    person, conf = match
                    result_lines.append(
                        f"人脸{i+1}: {person['姓名']} (年龄:{person['年龄']} 性别:{person['性别']} 学号:{person['学号']})\n"
                        f"       置信度:{conf:.2f}"
                    )
                else:
                    result_lines.append(f"人脸{i+1}: 未知人员")
            except Exception as e:
                result_lines.append(f"人脸{i+1}: 特征提取失败")
        result_text = "\n\n".join(result_lines)
        return frame_with_boxes, result_text

    def _find_best_match(self, query_emb, threshold=0.8):
        if not self.db_cache:
            return None
        min_dist = float('inf')
        best = None
        for item in self.db_cache:
            dist = np.linalg.norm(query_emb - item['embedding'])
            if dist < min_dist and dist < threshold:
                min_dist = dist
                best = item
        if best:
            confidence = 1 - min_dist / threshold
            return best, confidence
        return None

    def stop(self):
        self.running = False
        self.wait()


# ---------- 主窗口 ----------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.db = MySqlite()
        self._create_table_if_not_exists()

        # 加载模型
        yolo_path = r'C:\Users\Lenovo\Desktop\HQYJ\Facial_Recognition\my_yolo\runs\detect\train\weights\best.pt'
        self.face_net = FaceNet(yolo_path)

        # 特征缓存
        self.face_cache = []  # 每个元素为 {'姓名':..., '年龄':..., ... , 'embedding': np.ndarray}

        # 摄像头线程（单例）
        self.camera_thread = CameraThread(0)
        self.camera_thread.set_face_net(self.face_net)
        self.camera_thread.frame_processed.connect(self.on_camera_frame)
        self.camera_thread.status_signal.connect(lambda msg: QMessageBox.warning(self, "摄像头", msg))

        # 当前捕获的人脸（用于录入）
        self.current_faces = []   # 人脸图像列表

        # 初始化信号
        self._init_signals()

        # 加载数据库数据到缓存
        self._refresh_cache()

        # 加载表格
        self._load_table_data()

    # ---------- 数据库辅助 ----------
    def _create_table_if_not_exists(self):
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
        self.db.operation_sql(sql)

    def _refresh_cache(self):
        """从数据库加载所有人脸特征到缓存"""
        sql = "SELECT ID, 姓名, 年龄, 性别, 学号, 录入时间, 照片 FROM student_info"
        rows = self.db.operation_sql(sql)
        if not rows or rows is True:
            self.face_cache = []
            return

        cache = []
        for row in rows:
            try:
                blob = row[6]
                if not blob:
                    continue
                nparr = np.frombuffer(blob, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                emb = self.face_net.get_embedding(img)
                cache.append({
                    'ID': row[0],
                    '姓名': row[1],
                    '年龄': row[2],
                    '性别': row[3],
                    '学号': row[4],
                    '录入时间': row[5],
                    'embedding': emb
                })
            except Exception as e:
                print(f"缓存加载错误: {e}")
        self.face_cache = cache
        self.camera_thread.set_db_cache(self.face_cache)  # 同步到摄像头线程

    # ---------- UI 信号连接 ----------
    def _init_signals(self):
        # 页面切换
        self.ui.pushButton_sb_3.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(0))
        self.ui.pushButton_25.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(1))
        self.ui.pushButton_gl_3.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(2))
        self.ui.pushButton_lr_3.clicked.connect(self.close)

        # 录入页面
        self.ui.pushButton_14.clicked.connect(self._upload_for_enroll)
        self.ui.pushButton_15.clicked.connect(self._start_camera_enroll)
        self.ui.pushButton_16.clicked.connect(self._capture_face)
        self.ui.pushButton_17.clicked.connect(self._stop_camera)
        self.ui.pushButton_26.clicked.connect(self._save_face_info)

        # 识别页面
        self.ui.pushButton_19.clicked.connect(self._upload_for_recognition)
        self.ui.pushButton_20.clicked.connect(self._start_camera_recognize)
        self.ui.pushButton_21.clicked.connect(self._stop_camera)

        # 数据库管理页面
        self.ui.pushButton_22.clicked.connect(self._search_db)
        self.ui.pushButton_23.clicked.connect(self._delete_record)
        self.ui.pushButton_24.clicked.connect(self._refresh_table)

    # ---------- 通用方法 ----------
    def _stop_camera(self):
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
        self.ui.label_15.clear()
        self.ui.label_30.clear()
        self.ui.label_24.clear()
        self.ui.plainTextEdit_2.clear()

    def on_camera_frame(self, img, result_text):
        """摄像头线程处理完一帧后调用，更新UI"""
        pix = self.face_net.pixmap_from_cv(img)
        current_page = self.ui.stackedWidget_2.currentIndex()
        if current_page == 0:  # 录入页面
            self.ui.label_15.setPixmap(pix.scaled(self.ui.label_15.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:  # 识别页面
            self.ui.label_24.setPixmap(pix.scaled(self.ui.label_24.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.ui.plainTextEdit_2.setPlainText(result_text)

    # ---------- 录入功能 ----------
    def _upload_for_enroll(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self.ui.lineEdit_7.setText(path)
        img = cv2.imread(path)
        if img is None:
            return
        # 显示原图
        self.ui.label_15.setPixmap(self.face_net.pixmap_from_cv(img).scaled(
            self.ui.label_15.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # 检测人脸
        boxes, faces = self.face_net.detect_faces(img)
        if faces:
            self.current_faces = faces
            self.ui.label_30.setPixmap(self.face_net.pixmap_from_cv(faces[0]).scaled(
                self.ui.label_30.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            QMessageBox.information(self, "成功", f"检测到 {len(faces)} 张人脸，已选择第一张")
        else:
            QMessageBox.warning(self, "警告", "未检测到人脸")

    def _start_camera_enroll(self):
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
        self.camera_thread.set_mode("enroll")
        self.camera_thread.start()

    def _capture_face(self):
        """拍照功能简化：提示使用上传图片"""
        QMessageBox.information(self, "提示", "请使用「上传图片」方式录入人脸")

    def _save_face_info(self):
        name = self.ui.lineEdit_13.text().strip()
        age = self.ui.lineEdit_14.text().strip()
        student_id = self.ui.lineEdit_15.text().strip()
        gender = "男" if self.ui.radioButton_5.isChecked() else "女" if self.ui.radioButton_6.isChecked() else ""

        if not all([name, age, student_id, gender]):
            QMessageBox.warning(self, "警告", "请填写完整信息")
            return
        try:
            age = int(age)
        except:
            QMessageBox.warning(self, "警告", "年龄必须为数字")
            return

        if not self.current_faces:
            QMessageBox.warning(self, "警告", "没有可保存的人脸，请先上传或拍照")
            return

        face_img = self.current_faces[0]
        try:
            # 仅用于校验，实际可以不用
            _ = self.face_net.get_embedding(face_img)
        except:
            QMessageBox.critical(self, "错误", "特征提取失败")
            return

        # 保存到数据库
        _, img_encoded = cv2.imencode('.jpg', face_img)
        photo_bytes = img_encoded.tobytes()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = "INSERT INTO student_info (姓名,年龄,性别,学号,录入时间,照片) VALUES (?,?,?,?,?,?)"
        ok = self.db.operation_sql(sql, [name, age, gender, student_id, now, photo_bytes])
        if ok is True:
            QMessageBox.information(self, "成功", "保存成功")
            self._refresh_cache()          # 更新缓存
            self._clear_enroll_fields()
        else:
            QMessageBox.critical(self, "错误", f"保存失败: {ok}")

    def _clear_enroll_fields(self):
        self.ui.lineEdit_13.clear()
        self.ui.lineEdit_14.clear()
        self.ui.lineEdit_15.clear()
        self.ui.radioButton_5.setChecked(False)
        self.ui.radioButton_6.setChecked(False)
        self.ui.label_30.clear()
        self.current_faces = []

    # ---------- 识别功能 ----------
    def _upload_for_recognition(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self.ui.lineEdit_11.setText(path)
        img = cv2.imread(path)
        if img is None:
            return
        # 显示原图
        self.ui.label_24.setPixmap(self.face_net.pixmap_from_cv(img).scaled(
            self.ui.label_24.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # 检测人脸
        boxes, faces = self.face_net.detect_faces(img)
        if not faces:
            self.ui.plainTextEdit_2.setPlainText("未检测到人脸")
            return

        result_lines = []
        for i, face in enumerate(faces):
            try:
                emb = self.face_net.get_embedding(face)
                match = self._find_match_in_cache(emb)
                if match:
                    person, conf = match
                    result_lines.append(f"人脸{i+1}: {person['姓名']} (置信度:{conf:.2f})")
                else:
                    result_lines.append(f"人脸{i+1}: 未知人员")
            except:
                result_lines.append(f"人脸{i+1}: 特征提取失败")
        self.ui.plainTextEdit_2.setPlainText("\n".join(result_lines))

    def _find_match_in_cache(self, emb, threshold=0.8):
        if not self.face_cache:
            return None
        min_dist = float('inf')
        best = None
        for item in self.face_cache:
            dist = np.linalg.norm(emb - item['embedding'])
            if dist < min_dist and dist < threshold:
                min_dist = dist
                best = item
        if best:
            conf = 1 - min_dist / threshold
            return best, conf
        return None

    def _start_camera_recognize(self):
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
        self.camera_thread.set_mode("recognize")
        self.camera_thread.set_db_cache(self.face_cache)  # 确保最新缓存
        self.camera_thread.start()

    # ---------- 数据库管理 ----------
    def _search_db(self):
        keyword = self.ui.lineEdit_12.text().strip()
        if not keyword:
            QMessageBox.warning(self, "警告", "请输入关键词")
            return
        sql = "SELECT ID, 姓名, 年龄, 性别, 学号, 录入时间 FROM student_info WHERE 姓名 LIKE ? OR 学号 LIKE ?"
        param = f"%{keyword}%"
        rows = self.db.operation_sql(sql, [param, param])
        self._load_table_data(rows if rows else [])

    def _load_table_data(self, data=None):
        if data is None:
            sql = "SELECT ID, 姓名, 年龄, 性别, 学号, 录入时间 FROM student_info"
            data = self.db.operation_sql(sql) or []

        self.ui.tableView_2.setRowCount(0)
        self.ui.tableView_2.setColumnCount(6)
        self.ui.tableView_2.setHorizontalHeaderLabels(["ID", "姓名", "年龄", "性别", "学号", "录入时间"])
        self.ui.tableView_2.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for r, row in enumerate(data):
            self.ui.tableView_2.insertRow(r)
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val) if val is not None else "")
                item.setTextAlignment(Qt.AlignCenter)
                self.ui.tableView_2.setItem(r, c, item)

    def _delete_record(self):
        selected = self.ui.tableView_2.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(self, "警告", "请先选择要删除的行")
            return
        reply = QMessageBox.question(self, "确认", "确定删除选中记录？", QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        for idx in sorted(selected, key=lambda x: x.row(), reverse=True):
            row = idx.row()
            id_item = self.ui.tableView_2.item(row, 0)
            if id_item:
                self.db.operation_sql("DELETE FROM student_info WHERE ID=?", [id_item.text()])
        self._refresh_cache()
        self._load_table_data()

    def _refresh_table(self):
        self._load_table_data()

    def closeEvent(self, event):
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
        event.accept()


# ---------- 程序入口 ----------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())