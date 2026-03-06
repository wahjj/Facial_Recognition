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
    """SQLite数据库操作类，负责数据库连接和SQL操作"""
    def __init__(self):
        """初始化数据库连接"""
        try:
            # 创建SQLite数据库连接
            self.database = QSqlDatabase.addDatabase("QSQLITE")
            # 设置数据库文件名
            self.database.setDatabaseName("student.db")
            # 打开数据库连接
            self.database.open()
            # 创建查询对象
            self.query = QSqlQuery()
        except Exception as e:
            print(f"无法建立连接，error:{e}")

    def operation_sql(self, sql, params=None):
        """执行SQL操作，支持查询和增删改操作
        Args:
            sql: SQL语句
            params: 参数列表（用于预处理语句）
        Returns:
            查询返回结果列表，其他操作返回True表示成功，字符串表示错误
        """
        try:
            if params:
                # 如果有参数，使用预处理语句
                if not self.query.prepare(sql):
                    return self.query.lastError().text()
                # 逐个绑定参数值
                for param in params:
                    if isinstance(param, bytes):
                        # 处理字节数组类型参数
                        self.query.addBindValue(QByteArray(param))
                    else:
                        # 处理普通参数
                        self.query.addBindValue(param)
                if not self.query.exec_():
                    return self.query.lastError().text()
            else:
                # 直接执行SQL语句
                if not self.query.exec_(sql):
                    return self.query.lastError().text()
            # 判断是否为SELECT查询语句
            if sql.strip().upper().startswith("SELECT"):
                results = []
                # 获取查询结果的所有行
                while self.query.next():
                    row = []
                    # 获取每行的每个字段值
                    for i in range(self.query.record().count()):
                        value = self.query.value(i)
                        if isinstance(value, QByteArray):
                            # 转换字节数组
                            value = bytes(value)
                        row.append(value)
                    results.append(row)
                return results
            else:
                # 非查询操作返回成功标志
                return True
        except Exception as e:
            return str(e)


class FaceNet:
    """人脸识别核心类，包含YOLO检测和FaceNet特征提取功能"""
    def __init__(self, yolo_model_path):
        """初始化人脸识别模型
        Args:
            yolo_model_path: YOLO模型文件路径
        """
        # 加载FaceNet预训练模型，使用CASIA-WebFace数据集预训练
        self.facenet_model = InceptionResnetV1(pretrained='casia-webface').eval().to('cpu')
        # 加载YOLO检测模型
        self.yolo_model = YOLO(yolo_model_path)

    def preprocess_face_img(self, face_img):
        """预处理人脸图像为模型输入格式
        Args:
            face_img: 输入的人脸图像（BGR格式）
        Returns:
            预处理后的PyTorch张量
        """
        # 转换颜色空间从BGR到RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # 调整图像大小为160x160（FaceNet模型要求的输入尺寸）
        face_img = cv2.resize(face_img, (160, 160))
        # 转换为PyTorch张量，调整维度顺序(HWC -> CHW)，归一化到[0,1]
        face_tensor = (torch.tensor(face_img).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
        return face_tensor

    def facenet(self, face_img):
        """提取人脸特征向量
        Args:
            face_img: 输入的人脸图像
        Returns:
            归一化后的人脸特征向量（numpy数组）
        """
        # 预处理图像
        face_tensor = self.preprocess_face_img(face_img=face_img)
        # 关闭梯度计算，提高推理效率
        with torch.no_grad():
            # 提取原始特征向量
            face_embedding = self.facenet_model(face_tensor)
            # 计算L2范数（欧几里得范数）
            l2_norm = torch.norm(face_embedding, p=2, dim=1, keepdim=True)
            # L2归一化，使特征向量长度为1
            face_embedding_normalized = face_embedding.div(l2_norm)
        # 转换为numpy数组并移动到CPU
        return face_embedding_normalized.cpu().numpy()

    def getfacepos(self, img):
        """检测图像中的人脸位置并裁剪出人脸
        Args:
            img: 图像路径、目录路径或numpy数组
        Returns:
            包含所有人脸图像的列表
        """
        face_imgs = []
        if isinstance(img, str):
            img_path = Path(img)
            if img_path.is_dir():
                # 如果输入是目录，遍历目录下的所有jpg文件
                for image_path in img_path.glob('*.jpg'):
                    frame = cv2.imread(str(image_path))
                    # 使用YOLO模型预测人脸位置
                    results = self.yolo_model.predict(frame, conf=0.7, verbose=False)
                    boxes = results[0].boxes
                    for box in boxes:
                        # 获取边界框坐标并转换为整数
                        x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                        # 裁剪出人脸区域
                        face_img = frame[y0:y1, x0:x1]
                        face_imgs.append(face_img)
            elif img_path.is_file():
                # 如果输入是单个文件，直接处理
                frame = cv2.imread(str(img_path))
                results = self.yolo_model.predict(frame, conf=0.7, verbose=False)
                boxes = results[0].boxes
                for box in boxes:
                    x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                    face_img = frame[y0:y1, x0:x1]
                    face_imgs.append(face_img)
        elif isinstance(img, np.ndarray):
            # 如果输入是numpy数组（如摄像头帧），直接处理
            frame = img
            results = self.yolo_model.predict(frame, conf=0.7, verbose=False)
            boxes = results[0].boxes
            for box in boxes:
                x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                face_img = frame[y0:y1, x0:x1]
                face_imgs.append(face_img)
        return face_imgs

    def display_original_image(self, img):
        """将OpenCV图像转换为Qt显示格式
        Args:
            img: OpenCV格式的图像（BGR）
        Returns:
            QPixmap对象，用于Qt控件显示
        """
        # 转换颜色空间从BGR到RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 获取图像属性
        h, w, ch = rgb_img.shape
        # 计算每行字节数
        bytes_per_line = ch * w
        # 创建QImage对象
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 转换为QPixmap
        pixmap = QPixmap.fromImage(qimg)
        return pixmap


class MainWindow(QWidget):
    """主窗口类，继承自QWidget，包含人脸识别系统的完整界面和功能"""
    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        # 初始化UI界面
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # 初始化数据库连接
        self.db = MySqlite()
        # 摄像头对象
        self.cap = None
        # 定时器对象
        self.timer = QTimer()
        # 当前摄像头捕获的图像
        self.current_image = None
        # 当前检测到的人脸列表
        self.current_faces = []

        # YOLO模型路径
        self.yolo_path = r'/my_yolo/runs/detect/train/weights/best.pt'
        # 初始化人脸识别引擎
        self.face_net = FaceNet(self.yolo_path)

        # 初始化信号连接
        self.init_signals()
        # 创建数据库表
        self.create_table_if_not_exists()
        # 加载表格数据
        self.load_table_data()

    def init_signals(self):
        """初始化界面信号连接"""
        # 左侧导航按钮连接到不同页面
        self.ui.pushButton_sb_3.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(0))  # 上传图片页
        self.ui.pushButton_25.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(1))   # 摄像头录入页
        self.ui.pushButton_gl_3.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(2)) # 数据管理页
        self.ui.pushButton_lr_3.clicked.connect(self.close)  # 关闭程序

        # 录入页面按钮信号连接
        self.ui.pushButton_14.clicked.connect(self.upload_image_for_enrollment)   # 上传图片
        self.ui.pushButton_15.clicked.connect(self.start_camera_enrollment)       # 开启摄像头
        self.ui.pushButton_16.clicked.connect(self.capture_face_enrollment)       # 捕获人脸
        self.ui.pushButton_17.clicked.connect(self.stop_camera_enrollment)        # 关闭摄像头
        self.ui.pushButton_26.clicked.connect(self.save_face_info)               # 保存信息

        # 识别页面按钮信号连接
        self.ui.pushButton_19.clicked.connect(self.upload_image_for_recognition)  # 上传识别图片
        self.ui.pushButton_20.clicked.connect(self.start_camera_recognition)      # 开始识别
        self.ui.pushButton_21.clicked.connect(self.stop_camera_recognition)       # 停止识别

        # 管理页面按钮信号连接
        self.ui.pushButton_22.clicked.connect(self.search_database)              # 搜索
        self.ui.pushButton_23.clicked.connect(self.delete_record)                # 删除
        self.ui.pushButton_24.clicked.connect(self.refresh_table)                # 刷新

        # 定时器信号连接
        self.timer.timeout.connect(self.update_frame)

    def create_table_if_not_exists(self):
        """创建学生信息表（如果不存在）"""
        sql = """
        CREATE TABLE IF NOT EXISTS student_info(
            ID INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
            姓名 TEXT,                             -- 姓名
            年龄 INTEGER,                          -- 年龄
            性别 TEXT,                             -- 性别
            学号 TEXT,                             -- 学号
            录入时间 TEXT,                         -- 录入时间
            照片 BLOB                            -- 照片二进制数据
        )
        """
        result = self.db.operation_sql(sql)
        if result is True:
            print("数据库表创建成功")
        else:
            print(f"数据库表创建失败: {result}")

    def set_pixmap_to_label(self, label, img, width=None, height=None):
        """将OpenCV图像设置到QLabel控件显示
        Args:
            label: QLabel控件对象
            img: OpenCV格式的图像
            width: 显示宽度，默认为label宽度
            height: 显示高度，默认为label高度
        """
        if width is None:
            width = label.width()
        if height is None:
            height = label.height()
        # 转换图像格式并缩放到指定尺寸
        pixmap = self.face_net.display_original_image(img)
        label.setPixmap(pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def get_input_values(self):
        """获取录入表单的输入值
        Returns:
            (姓名, 年龄, 学号, 性别)元组
        """
        name = self.ui.lineEdit_13.text().strip()      # 获取姓名
        age = self.ui.lineEdit_14.text().strip()       # 获取年龄
        student_id = self.ui.lineEdit_15.text().strip() # 获取学号
        # 获取性别选择（单选按钮）
        gender = "男" if self.ui.radioButton_5.isChecked() else ("女" if self.ui.radioButton_6.isChecked() else "")
        return name, age, student_id, gender

    def validate_inputs(self, name, age, student_id, gender):
        """验证输入字段的合法性
        Args:
            name, age, student_id, gender: 输入的各个字段值
        Returns:
            bool: 验证通过返回True，否则显示警告并返回False
        """
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
        """重置录入界面的所有字段"""
        self.ui.lineEdit_13.clear()          # 清空姓名
        self.ui.lineEdit_14.clear()          # 清空年龄
        self.ui.lineEdit_15.clear()          # 清空学号
        self.ui.radioButton_5.setChecked(False)  # 取消男性选择
        self.ui.radioButton_6.setChecked(False)  # 取消女性选择
        self.ui.label_30.clear()             # 清空人脸预览
        self.current_faces = []              # 清空人脸列表

    def upload_image_for_enrollment(self):
        """上传图片进行人脸录入"""
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # 设置文件路径显示
            self.ui.lineEdit_7.setText(file_path)
            # 读取原始图像
            original_img = cv2.imread(file_path)
            if original_img is not None:
                # 显示原始图像
                self.set_pixmap_to_label(self.ui.label_15, original_img)
            # 检测图像中的人脸
            face_imgs = self.face_net.getfacepos(file_path)
            if face_imgs:
                # 保存检测到的人脸
                self.current_faces = face_imgs
                # 显示第一个人脸
                self.set_pixmap_to_label(self.ui.label_30, face_imgs[0])
                QMessageBox.information(self, "成功", f"检测到 {len(face_imgs)} 张人脸\n已自动选择第一张人脸")
            else:
                QMessageBox.warning(self, "警告", "未检测到人脸，请更换图片！")

    def start_camera_enrollment(self):
        """开启摄像头进行人脸录入"""
        if self.cap is not None and self.cap.isOpened():
            QMessageBox.warning(self, "警告", "摄像头已在运行中！")
            return
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头！请检查设备连接。")
            return
        # 启动定时器，开始捕获视频帧
        self.timer.start(30)
        self.current_faces = []

    def capture_face_enrollment(self):
        """从当前摄像头画面捕获人脸"""
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "警告", "请先开启摄像头！")
            return
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "当前没有可用的摄像头画面！")
            return
        # 从当前画面检测人脸
        face_imgs = self.face_net.getfacepos(self.current_image)
        if face_imgs:
            self.current_faces = face_imgs
            self.set_pixmap_to_label(self.ui.label_30, face_imgs[0])
            QMessageBox.information(self, "成功", f"检测到 {len(face_imgs)} 张人脸\n已自动选择第一张人脸")
        else:
            QMessageBox.warning(self, "警告", "当前画面未检测到人脸！")

    def stop_camera_enrollment(self):
        """停止摄像头录入"""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        # 清空显示
        self.ui.label_15.clear()
        self.ui.label_30.clear()

    def update_frame(self):
        """更新摄像头帧（用于录入模式）"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_image = frame.copy()
                # 在帧上绘制人脸检测框
                detected_frame = self.draw_face_boxes(frame)
                # 显示处理后的帧
                self.set_pixmap_to_label(self.ui.label_15, detected_frame)

    def draw_face_boxes(self, frame):
        """在图像上绘制人脸检测框
        Args:
            frame: 输入图像
        Returns:
            绘制了检测框的图像
        """
        processed_frame = frame.copy()
        # 使用YOLO模型检测人脸
        results = self.face_net.yolo_model.predict(processed_frame, conf=0.7, verbose=False)
        boxes = results[0].boxes
        for box in boxes:
            # 获取边界框坐标
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            # 绘制绿色矩形框
            cv2.rectangle(processed_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return processed_frame

    def save_face_info(self):
        """保存人脸信息到数据库"""
        # 获取输入值
        name, age, student_id, gender = self.get_input_values()
        # 验证输入
        if not self.validate_inputs(name, age, student_id, gender):
            return
        if not self.current_faces:
            QMessageBox.warning(self, "警告", "没有检测到人脸！")
            return
        try:
            # 提取第一个人脸的特征
            face_img = self.current_faces[0]
            embedding = self.face_net.facenet(face_img)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"人脸特征提取失败：{str(e)}")
            return
        # 将人脸图像编码为JPEG格式的字节数据
        _, img_encoded = cv2.imencode('.jpg', face_img)
        photo_bytes = img_encoded.tobytes()
        # 获取当前时间
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 插入数据库的SQL语句
        sql = "INSERT INTO student_info (姓名,年龄,性别,学号,录入时间,照片) VALUES (?, ?, ?, ?, ?, ?)"
        result = self.db.operation_sql(sql, [name, int(age), gender, student_id, current_time, photo_bytes])
        if result is True:
            QMessageBox.information(self, "成功", "人脸信息保存成功！")
            self.reset_enrollment_fields()
        else:
            QMessageBox.critical(self, "错误", f"保存失败：{result}")

    def upload_image_for_recognition(self):
        """上传图片进行人脸识别"""
        # 选择识别图片
        file_path, _ = QFileDialog.getOpenFileName(self, "选择识别图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # 显示文件路径
            self.ui.lineEdit_11.setText(file_path)
            # 读取并显示原始图像
            original_img = cv2.imread(file_path)
            if original_img is not None:
                self.set_pixmap_to_label(self.ui.label_24, original_img)
                # 检测图像中的人脸
                face_imgs = self.face_net.getfacepos(file_path)
                if face_imgs:
                    recognition_results = []
                    for i, face_img in enumerate(face_imgs):
                        try:
                            # 提取人脸特征
                            embedding = self.face_net.facenet(face_img)
                            # 在数据库中查找匹配
                            match_result = self.find_matches(embedding)
                            if match_result:
                                name, confidence = match_result
                                recognition_results.append(f"人脸{i+1}: {name} (置信度: {confidence:.2f})")
                            else:
                                recognition_results.append(f"人脸{i+1}: 未知人员 (置信度: 0.00)")
                        except Exception as e:
                            recognition_results.append(f"人脸{i+1}: 特征提取失败 ({str(e)})")
                    # 显示识别结果
                    result_text = "\n".join(recognition_results)
                    self.ui.plainTextEdit_2.setPlainText(result_text)
                else:
                    QMessageBox.warning(self, "警告", "未检测到人脸！")
                    self.ui.plainTextEdit_2.setPlainText("未检测到人脸！")
            else:
                QMessageBox.critical(self, "错误", "无法读取图片文件！")
                self.ui.plainTextEdit_2.setPlainText("无法读取图片文件！")

    def find_matches(self, query_embedding, threshold=1.0):
        """在数据库中查找最匹配的人脸
        Args:
            query_embedding: 查询人脸的特征向量
            threshold: 匹配阈值
        Returns:
            (人员信息, 置信度) 或 None（无匹配）
        """
        # 查询数据库中的所有记录
        sql = "SELECT ID, 姓名, 年龄, 性别, 学号, 录入时间, 照片 FROM student_info"
        result = self.db.operation_sql(sql)
        if not result or result is True:
            return None

        min_distance = float('inf')  # 最小距离
        best_match = None           # 最佳匹配结果

        for row in result:
            blob_data = row[6]  # 获取照片数据
            if blob_data:
                # 解码照片数据
                nparr = np.frombuffer(blob_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    try:
                        # 提取数据库中照片的特征
                        db_embedding = self.face_net.facenet(img)
                        # 计算特征向量间的欧几里得距离
                        distance = np.linalg.norm(query_embedding - db_embedding)
                        # 更新最小距离和最佳匹配
                        if distance < min_distance and distance < threshold:
                            min_distance = distance
                            # 构造人员信息字典
                            person_info = {'姓名': row[1], '年龄': row[2], '性别': row[3], '学号': row[4], '录入时间': row[5]}
                            best_match = (person_info, 1 - distance)  # 置信度为1-距离
                    except Exception as e:
                        print(f"特征提取错误: {str(e)}")
                        continue
        return best_match if best_match else None

    def start_camera_recognition(self):
        """开启摄像头进行实时人脸识别"""
        if self.cap is not None and self.cap.isOpened():
            QMessageBox.warning(self, "警告", "摄像头已在运行中！")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头！请检查设备连接。")
            return

        # 识别锁定标志，防止结果频繁变化
        self.recognition_locked = False
        self.locked_result = ""

        # 断开之前的连接，连接新的识别函数
        self.timer.timeout.disconnect()
        self.timer.timeout.connect(self.recognize_from_camera)
        self.timer.start(30)

    def recognize_from_camera(self):
        """从摄像头实时识别人脸"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if not self.recognition_locked:
                    # 进行人脸识别
                    result_frame, recognition_result = self.recognize_face_in_frame(frame)

                    # 如果识别到有效结果，锁定显示
                    if recognition_result and "没有该人员信息" not in recognition_result and recognition_result.strip() != "":
                        self.recognition_locked = True
                        self.locked_result = recognition_result
                        self.ui.plainTextEdit_2.setPlainText(recognition_result)
                    else:
                        # 否则继续更新显示
                        self.set_pixmap_to_label(self.ui.label_24, result_frame)
                        self.ui.plainTextEdit_2.setPlainText(recognition_result)
                else:
                    # 锁定状态下继续显示处理后的帧，但保持锁定的结果
                    result_frame, _ = self.recognize_face_in_frame(frame)
                    self.set_pixmap_to_label(self.ui.label_24, result_frame)
                    self.ui.plainTextEdit_2.setPlainText(self.locked_result)

    def recognize_face_in_frame(self, frame):
        """在单帧中识别人脸
        Args:
            frame: 输入的视频帧
        Returns:
            (处理后的帧, 识别结果文本)
        """
        processed_frame = frame.copy()
        # 检测帧中的人脸
        results = self.face_net.yolo_model.predict(processed_frame, conf=0.7, verbose=False)
        boxes = results[0].boxes
        recognition_result = ""
        for i, box in enumerate(boxes):
            # 获取人脸位置
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            face_img = processed_frame[y0:y1, x0:x1]
            try:
                # 提取人脸特征
                embedding = self.face_net.facenet(face_img)
                # 查找匹配
                match_result = self.find_matches(embedding)
                if match_result:
                    person_info, confidence = match_result
                    name = person_info['姓名']
                    age = person_info['年龄']
                    gender = person_info['性别']
                    student_id = person_info['学号']
                    record_time = person_info['录入时间']
                    # 构造详细识别结果
                    recognition_result += f"人脸{i+1}:\n姓名: {name}\n年龄: {age}\n性别: {gender}\n学号: {student_id}\n录入时间: {record_time}\n置信度: {confidence:.2f}\n\n"
                else:
                    recognition_result += f"人脸{i+1}: 没有该人员信息\n\n"
            except Exception as e:
                recognition_result += f"人脸{i+1}: 特征提取失败 ({str(e)})\n\n"

            # 在帧上绘制检测框
            cv2.rectangle(processed_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        return processed_frame, recognition_result

    def stop_camera_recognition(self):
        """停止摄像头人脸识别"""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        # 清空显示
        self.ui.label_24.clear()
        self.ui.plainTextEdit_2.clear()

    def search_database(self):
        """搜索数据库中的记录"""
        keyword = self.ui.lineEdit_12.text().strip()
        if not keyword:
            QMessageBox.warning(self, "警告", "请输入查询关键词！")
            return

        # 模糊查询SQL语句
        sql = "SELECT ID, 姓名, 年龄, 性别, 学号, 录入时间 FROM student_info WHERE 姓名 LIKE ? OR 学号 LIKE ?"
        param = f"%{keyword}%"
        result = self.db.operation_sql(sql, [param, param])

        results = []
        if result and result is not True:
            results = result

        # 加载搜索结果到表格
        self.load_table_data(results)

    def load_table_data(self, data=None):
        """加载表格数据
        Args:
            data: 要显示的数据列表，None时显示所有数据
        """
        # 清空现有表格行
        self.ui.tableView_2.setRowCount(0)

        # 设置表头
        headers = ["ID", "姓名", "年龄", "性别", "学号", "录入时间"]
        self.ui.tableView_2.setColumnCount(len(headers))
        self.ui.tableView_2.setHorizontalHeaderLabels(headers)

        # 设置列宽自适应
        header = self.ui.tableView_2.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        if data is None:
            # 如果没有传入数据，查询数据库获取所有记录
            sql = "SELECT ID, 姓名, 年龄, 性别, 学号, 录入时间 FROM student_info"
            result = self.db.operation_sql(sql)
            if result and result is not True:
                data = result
            else:
                data = []

        if data:
            # 设置表格行数
            self.ui.tableView_2.setRowCount(len(data))
            for row_idx, row_data in enumerate(data):
                for col_idx, data in enumerate(row_data):
                    if isinstance(data, bytes):
                        data = "BLOB数据"
                    # 创建表格项并居中显示
                    item = QTableWidgetItem(str(data) if data is not None else "")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.ui.tableView_2.setItem(row_idx, col_idx, item)

    def delete_record(self):
        """删除选中的记录"""
        selected_rows = self.ui.tableView_2.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请先选择要删除的记录！")
            return

        # 确认删除对话框
        reply = QMessageBox.question(self, '确认删除', '确定要删除选中的记录吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 从后往前删除，避免索引变化问题
            for index in sorted(selected_rows, key=lambda x: x.row(), reverse=True):
                row = index.row()
                id_item = self.ui.tableView_2.item(row, 0)
                if id_item:
                    record_id = id_item.text()
                    sql = "DELETE FROM student_info WHERE ID = ?"
                    result = self.db.operation_sql(sql, [record_id])
                    if result is not True:
                        QMessageBox.critical(self, "错误", f"删除记录 ID={record_id} 失败")

            # 重新加载表格数据
            self.load_table_data()
            QMessageBox.information(self, "成功", "记录删除成功！")

    def refresh_table(self):
        """刷新表格显示"""
        self.load_table_data()

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        if self.cap is not None:
            self.cap.release()
        if self.timer.isActive():
            self.timer.stop()
        event.accept()


if __name__ == '__main__':
    # 初始化数据库
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

    # 启动应用程序
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())