# 导入系统相关的模块
# sys 模块提供对 Python 解释器使用的一些变量的访问
import sys
# os 模块提供与操作系统交互的功能
import os
# 导入 OpenCV 库，用于图像处理和摄像头操作
import cv2
# 导入 PyTorch 深度学习框架
import torch
# 导入 NumPy 库，用于数值计算和数组操作
import numpy as np
# 从 datetime 模块导入 datetime 类，用于获取当前时间
from datetime import datetime
# 从 PyQt5 的 QtWidgets 模块导入 GUI 组件
# QApplication: 应用程序管理
# QWidget: 窗口部件基类
# QMessageBox: 消息对话框
# QFileDialog: 文件选择对话框
# QTableWidgetItem: 表格项
# QHeaderView: 表头视图
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QFileDialog, QTableWidgetItem, QHeaderView
# 从 PyQt5 的 QtCore 模块导入核心功能
# QTimer: 定时器
# Qt: Qt 命名空间
# QByteArray: 字节数组
# QThread: 线程类
# pyqtSignal: PyQt 信号
from PyQt5.QtCore import QTimer, Qt, QByteArray, QThread, pyqtSignal
# 从 PyQt5 的 QtGui 模块导入图形界面相关类
# QImage: 图像数据格式
# QPixmap: 用于在界面上显示图像的类
from PyQt5.QtGui import QImage, QPixmap
# 从 face_ui 模块导入 Ui_Form 类（UI 界面设计）
from face_ui import Ui_Form
# 从 sqlite_db 模块导入 MySqlite 类（数据库操作）
from sqlite_db import MySqlite
# 从 facenet_model 模块导入 FaceNet 类（人脸识别模型）
from facenet_model import FaceNet

# 移除环境变量中的 QT_QPA_PLATFORM_PLUGIN_PATH
# 这是为了解决某些平台上 Qt 插件路径问题导致的启动错误
# pop() 方法删除指定的键，如果不存在也不会报错
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


# 定义 CameraThread 类，继承自 QThread
# 用于在独立线程中处理摄像头画面，避免阻塞主界面
class CameraThread(QThread):
    # 定义自定义信号 frame_processed
    # 该信号用于向主线程发送处理后的帧数据和识别结果
    # object 参数类型表示可以传递任何对象（这里是处理后的图像帧）
    # str 参数类型表示传递字符串（这里是识别结果文本）
    frame_processed = pyqtSignal(object, str)
    # 定义自定义信号 status_updated
    # 该信号用于向主线程发送状态更新信息
    # str 参数类型表示传递状态消息
    status_updated = pyqtSignal(str)

    # 类的初始化方法
    def __init__(self, parent=None):
        # 调用父类 QThread 的初始化方法
        # parent 参数指定父对象，用于内存管理
        super().__init__(parent)
        # 初始化运行状态标志，默认为 False
        # 用于控制线程的运行和停止
        self.running = False
        # 初始化工作模式变量
        # 'enrollment' 表示人脸录入模式
        # 'recognition' 表示人脸识别模式
        self.mode = None  # 'enrollment' or 'recognition'
        # 初始化 FaceNet 模型引用
        # 用于人脸检测和特征提取
        self.face_net = None
        # 初始化摄像头捕获对象
        # 用于从摄像头读取视频帧
        self.cap = None
        # 初始化人脸缓存列表
        # 存储已录入人员的特征向量和信息
        self.face_cache = []
        # 初始化识别阈值，默认为 0.8
        # 用于判断两个人脸是否属于同一个人
        # 距离小于此阈值才认为是匹配
        self.threshold = 0.8

    # 定义线程配置方法
    # 用于在线程启动前设置必要的参数
    def setup(self, cap, face_net, mode, face_cache, threshold):
        # 保存摄像头捕获对象
        self.cap = cap
        # 保存 FaceNet 模型引用
        self.face_net = face_net
        # 保存工作模式
        self.mode = mode
        # 保存人脸缓存数据
        self.face_cache = face_cache
        # 保存识别阈值
        self.threshold = threshold

    # 定义线程运行方法
    # 当调用 start() 方法时，run() 方法会在新的线程中执行
    def run(self):
        # 设置运行状态为 True，开始循环处理
        self.running = True
        # 进入无限循环，持续处理摄像头帧
        while self.running:
            # 检查摄像头对象是否存在且已打开
            # isOpened() 方法检查摄像头是否可用
            if self.cap is not None and self.cap.isOpened():
                # 从摄像头读取一帧图像
                # ret: 布尔值，表示是否成功读取
                # frame: 读取到的图像帧（numpy 数组）
                ret, frame = self.cap.read()
                # 如果成功读取帧
                if ret:
                    # 判断当前工作模式是否为识别模式
                    if self.mode == 'recognition':
                        # 调用人脸识别方法处理当前帧
                        # processed_frame: 绘制了检测框和识别结果的图像
                        # recognition_result: 识别结果的文本描述
                        processed_frame, recognition_result = self.recognize_face_in_frame(frame)
                        # 通过信号发送处理后的帧和识别结果到主线程
                        # emit() 方法触发信号
                        self.frame_processed.emit(processed_frame, recognition_result)
                    # 判断当前工作模式是否为录入模式
                    elif self.mode == 'enrollment':
                        # 调用绘制人脸框方法处理当前帧
                        # 只绘制检测框，不进行识别
                        processed_frame = self.draw_face_boxes(frame)
                        # 通过信号发送处理后的帧到主线程（空字符串作为第二个参数）
                        self.frame_processed.emit(processed_frame, "")
                # 如果读取失败
                else:
                    # 通过状态更新信号发送错误信息
                    self.status_updated.emit("摄像头读取失败")
                    # 跳出循环，结束线程
                    break
            # 如果摄像头未打开
            else:
                # 通过状态更新信号发送警告信息
                self.status_updated.emit("摄像头未打开")
                # 跳出循环，结束线程
                break

    # 定义线程停止方法
    # 用于安全地停止线程运行
    def stop(self):
        # 设置运行状态为 False
        # 这会导致 run() 方法中的 while 循环退出
        self.running = False

    # 定义绘制人脸检测框的方法
    # 用于在录入模式下显示检测到的人脸位置
    def draw_face_boxes(self, frame):
        """在帧上绘制人脸检测框"""
        # 复制输入帧，避免修改原始数据
        # copy() 方法创建图像的深拷贝
        processed_frame = frame.copy()
        # 使用 YOLO 模型进行人脸检测
        # predict() 方法执行目标检测
        # conf=0.7 设置置信度阈值为 0.7
        # verbose=False 关闭控制台输出
        results = self.face_net.yolo_model.predict(processed_frame, conf=0.7, verbose=False)
        # 获取检测结果中的所有检测框
        # boxes 属性包含所有检测到的人脸框信息
        boxes = results[0].boxes
        # 遍历所有检测框
        for box in boxes:
            # 提取检测框的坐标
            # xyxy[0] 获取第一个框的左上角和右下角坐标 (x0, y0, x1, y1)
            # tolist() 将张量转换为列表
            # map(int, ...) 将所有坐标转换为整数
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            # 在图像上绘制绿色矩形框
            # cv2.rectangle() 绘制矩形
            # (x0, y0) 左上角坐标
            # (x1, y1) 右下角坐标
            # (0, 255, 0) 颜色（BGR 格式，绿色）
            # 2 线条宽度（像素）
            cv2.rectangle(processed_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        # 返回绘制了检测框的图像
        return processed_frame

    # 定义人脸识别方法
    # 用于在识别模式下处理每一帧图像
    def recognize_face_in_frame(self, frame):
        """识别人脸帧"""
        # 复制输入帧，避免修改原始数据
        processed_frame = frame.copy()
        # 使用 YOLO 模型进行人脸检测
        results = self.face_net.yolo_model.predict(processed_frame, conf=0.7, verbose=False)
        # 获取所有检测框
        boxes = results[0].boxes
        # 初始化识别结果字符串为空
        recognition_result = ""

        # 遍历所有检测框，i 是索引，box 是当前检测框对象
        for i, box in enumerate(boxes):
            # 提取检测框坐标
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            # 根据检测框裁剪出人脸区域
            # frame[y0:y1, x0:x1] 使用 numpy 切片操作
            face_img = processed_frame[y0:y1, x0:x1]

            # 尝试进行人脸识别
            try:
                # 使用 FaceNet 提取人脸特征向量
                # facenet() 方法返回 512 维的特征向量
                embedding = self.face_net.facenet(face_img)
                # 在缓存中查找最相似的人脸
                # find_matches() 方法返回匹配的人员信息和置信度
                match_result = self.find_matches(embedding)

                # 如果找到匹配的结果
                if match_result:
                    # 解包匹配结果
                    # person_info: 包含人员信息的字典
                    # confidence: 匹配置信度（0-1 之间）
                    person_info, confidence = match_result
                    # 从人员信息中提取各个字段
                    name = person_info['姓名']
                    age = person_info['年龄']
                    gender = person_info['性别']
                    student_id = person_info['学号']
                    record_time = person_info['录入时间']

                    # 格式化识别结果字符串
                    # f-string 格式化输出，:.2f 保留两位小数
                    recognition_result += f"人脸{i+1}:\n姓名：{name}\n年龄：{age}\n性别：{gender}\n学号：{student_id}\n录入时间：{record_time}\n置信度：{confidence:.2f}\n\n"
                # 如果没有找到匹配的结果
                else:
                    # 添加未知人员提示到识别结果
                    recognition_result += f"人脸{i+1}: 没有该人员信息\n\n"
            # 捕获并处理异常
            except Exception as e:
                # 添加错误信息到识别结果
                recognition_result += f"人脸{i+1}: 特征提取失败 ({str(e)})\n\n"

            # 在图像上绘制绿色矩形框标记人脸位置
            cv2.rectangle(processed_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # 返回处理后的帧和识别结果
        return processed_frame, recognition_result

    # 定义查找最相似人脸的方法
    # 使用欧氏距离计算特征向量之间的相似度
    def find_matches(self, query_embedding):
        """查找数据库中最相似的人脸"""
        # 初始化最小距离为正无穷大
        # float('inf') 表示无穷大的浮点数
        min_distance = float('inf')
        # 初始化最佳匹配为 None
        best_match = None

        # 遍历人脸缓存中的每个人脸数据
        for person_data in self.face_cache:
            # 获取数据库中存储的人脸特征向量
            db_embedding = person_data['embedding']
            # 计算查询特征向量与数据库特征向量之间的欧氏距离
            # np.linalg.norm() 计算向量的范数（这里是 L2 范数/欧氏距离）
            distance = np.linalg.norm(query_embedding - db_embedding)

            # 如果当前距离小于已知最小距离，并且小于阈值
            if distance < min_distance and distance < self.threshold:
                # 更新最小距离
                min_distance = distance
                # 计算置信度
                # 置信度 = 1 - (距离 / 阈值)
                # 距离越小，置信度越高
                confidence = 1 - (distance / self.threshold)
                # 保存最佳匹配结果（人员信息字典和置信度）
                best_match = (person_data['info'], confidence)

        # 返回最佳匹配结果
        # 如果没有找到匹配，返回 None
        return best_match


# 定义 MainWindow 类，继承自 QWidget
# 这是应用程序的主窗口类
class MainWindow(QWidget):
    # 类的初始化方法
    def __init__(self):
        # 调用父类 QWidget 的初始化方法
        super().__init__()
        # 创建 UI 界面对象
        # Ui_Form 是从 face_ui 模块导入的界面类
        self.ui = Ui_Form()
        # 调用 setupUi() 方法初始化界面布局
        # 这会创建所有的按钮、标签、输入框等控件
        self.ui.setupUi(self)

        # 初始化数据库连接
        # MySqlite() 创建 SQLite 数据库操作对象
        self.db = MySqlite()

        # 初始化摄像头相关变量
        # cap 用于存储摄像头捕获对象
        self.cap = None
        # camera_thread 用于存储摄像头处理线程
        self.camera_thread = None
        # current_image 用于存储当前处理的图像帧
        self.current_image = None
        # current_faces 用于存储当前检测到的人脸图像列表
        self.current_faces = []

        # 设置 YOLO 模型权重文件的路径
        # r'' 表示原始字符串，避免转义字符问题
        self.yolo_path = r'C:\Users\Lenovo\Desktop\HQYJ\Facial_Recognition\my_yolo\runs\detect\train\weights\best.pt'
        # 创建 FaceNet 实例，加载 YOLO 模型
        self.face_net = FaceNet(self.yolo_path)

        # 初始化人脸缓存列表
        # 用于存储数据库中所有人的特征向量
        self.face_cache = []
        # 调用刷新缓存方法，从数据库加载人脸特征
        self._refresh_cache()

        # 调用信号槽初始化方法
        # 绑定所有按钮点击事件到对应的处理方法
        self.init_signals()

        # 调用创建数据库表方法
        # 如果表不存在则创建
        self.create_table_if_not_exists()

        # 调用加载表格数据方法
        # 在界面上显示数据库中的所有记录
        self.load_table_data()

    # 定义信号槽初始化方法
    # 将所有 UI 控件的信号连接到对应的方法
    def init_signals(self):
        """初始化信号槽连接"""
        # 左侧功能按钮的信号槽连接
        # connect() 方法将按钮的 clicked 信号连接到 lambda 函数
        # lambda 函数用于切换 stackedWidget_2 的页面索引
        # setCurrentIndex(0) 切换到第一个人脸录入页面
        self.ui.pushButton_sb_3.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(0))
        # setCurrentIndex(1) 切换到第二个人脸识别页面
        self.ui.pushButton_25.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(1))
        # setCurrentIndex(2) 切换到第三个数据库管理页面
        self.ui.pushButton_gl_3.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(2))
        # clicked.connect(self.close) 关闭窗口
        self.ui.pushButton_lr_3.clicked.connect(self.close)

        # 第一页：人脸信息录入功能的信号槽连接
        # 上传图片按钮连接到 upload_image_for_enrollment 方法
        self.ui.pushButton_14.clicked.connect(self.upload_image_for_enrollment)
        # 开启摄像头按钮连接到 start_camera_enrollment 方法
        self.ui.pushButton_15.clicked.connect(self.start_camera_enrollment)
        # 拍照按钮连接到 capture_face_enrollment 方法
        self.ui.pushButton_16.clicked.connect(self.capture_face_enrollment)
        # 关闭摄像头按钮连接到 stop_camera_enrollment 方法
        self.ui.pushButton_17.clicked.connect(self.stop_camera_enrollment)
        # 保存人脸信息按钮连接到 save_face_info 方法
        self.ui.pushButton_26.clicked.connect(self.save_face_info)

        # 第二页：人脸识别功能的信号槽连接
        # 上传图片识别按钮连接到 upload_image_for_recognition 方法
        self.ui.pushButton_19.clicked.connect(self.upload_image_for_recognition)
        # 开启摄像头识别按钮连接到 start_camera_recognition 方法
        self.ui.pushButton_20.clicked.connect(self.start_camera_recognition)
        # 关闭摄像头识别按钮连接到 stop_camera_recognition 方法
        self.ui.pushButton_21.clicked.connect(self.stop_camera_recognition)

        # 第三页：数据库管理功能的信号槽连接
        # 搜索按钮连接到 search_database 方法
        self.ui.pushButton_22.clicked.connect(self.search_database)
        # 删除记录按钮连接到 delete_record 方法
        self.ui.pushButton_23.clicked.connect(self.delete_record)
        # 刷新表格按钮连接到 refresh_table 方法
        self.ui.pushButton_24.clicked.connect(self.refresh_table)

    # 定义创建数据库表的方法
    # 如果表不存在则创建
    def create_table_if_not_exists(self):
        """创建数据库表"""
        # 定义创建表的 SQL 语句
        # CREATE TABLE IF NOT EXISTS: 如果表不存在才创建
        # ID: 主键，自动递增
        # 姓名：文本类型
        # 年龄：整数类型
        # 性别：文本类型
        # 学号：文本类型
        # 录入时间：文本类型
        # 照片：BLOB 类型（二进制数据，用于存储图片）
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
        # 执行 SQL 语句
        # operation_sql() 方法返回执行结果
        result = self.db.operation_sql(sql)
        # 如果执行成功
        if result:
            # 打印成功消息
            print("数据库表创建成功")
        # 如果执行失败
        else:
            # 打印失败消息
            print("数据库表创建失败")

    # 定义刷新人脸缓存的方法
    # 从数据库加载所有人员的特征向量
    def _refresh_cache(self):
        """刷新人脸缓存"""
        # 清空当前缓存列表
        self.face_cache = []
        # 定义查询所有记录的 SQL 语句
        sql = "SELECT ID, 姓名，年龄，性别，学号，录入时间，照片 FROM student_info"
        # 执行查询
        result = self.db.operation_sql(sql)

        # 如果查询结果不为空
        if result:
            # 遍历每一行数据
            for row in result:
                # 获取照片字段的数据（第 7 列，索引 6）
                blob_data = row[6]
                # 如果有照片数据
                if blob_data:
                    # 将字节数据转换为 numpy 数组
                    # np.frombuffer() 从字节缓冲区创建数组
                    # np.uint8 指定数据类型为 8 位无符号整数
                    nparr = np.frombuffer(blob_data, np.uint8)
                    # 使用 OpenCV 解码 numpy 数组为图像
                    # cv2.imdecode() 从内存缓冲区解码图像
                    # IMREAD_COLOR 以彩色模式读取
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    # 如果图像解码成功
                    if img is not None:
                        # 尝试提取人脸特征
                        try:
                            # 使用 FaceNet 提取人脸特征向量
                            embedding = self.face_net.facenet(img)
                            # 创建人员信息字典
                            person_info = {
                                # 从查询结果中提取各个字段
                                'ID': row[0],
                                '姓名': row[1],
                                '年龄': row[2],
                                '性别': row[3],
                                '学号': row[4],
                                '录入时间': row[5]
                            }
                            # 将特征向量和人员信息添加到缓存列表
                            self.face_cache.append({
                                'embedding': embedding,
                                'info': person_info
                            })
                        # 捕获并处理特征提取异常
                        except Exception as e:
                            # 打印错误信息
                            print(f"特征提取错误：{str(e)}")

    # 定义上传图片进行人脸录入的方法
    def upload_image_for_enrollment(self):
        """上传图片进行人脸录入"""
        # 打开文件选择对话框
        # getOpenFileName() 显示文件选择对话框
        # 返回值：file_path(选中的文件路径), _(选中的过滤器)
        # "选择图片": 对话框标题
        # "": 初始目录（空表示使用默认目录）
        # "Image Files (*.png *.jpg *.jpeg *.bmp)": 文件过滤器
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        # 如果用户选择了文件
        if file_path:
            # 在文件名输入框中显示选择的文件路径
            self.ui.lineEdit_7.setText(file_path)
            # 使用 OpenCV 读取图片
            original_img = cv2.imread(file_path)
            # 如果图片读取成功
            if original_img is not None:
                # 使用 FaceNet 的 display_original_image 方法转换图片格式
                # 将 OpenCV 的 BGR 格式转换为 PyQt 可显示的 RGB 格式
                pixmap1 = self.face_net.display_original_image(original_img)
                # 在 label_15 上显示原始图片
                # scaled() 方法缩放图片以适应 label 大小
                # KeepAspectRatio 保持宽高比
                # SmoothTransformation 使用平滑变换提高质量
                self.ui.label_15.setPixmap(pixmap1.scaled(
                    self.ui.label_15.width(), self.ui.label_15.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))

            # 使用 FaceNet 检测图片中的人脸
            # getfacepos() 方法返回所有检测到的人脸图像列表
            face_imgs = self.face_net.getfacepos(file_path)
            # 如果检测到人脸
            if face_imgs:
                # 保存检测到的人脸到当前人脸变量
                self.current_faces = face_imgs
                # 显示第一张检测到的人脸
                # display_original_image() 转换图片格式
                pixmap2 = self.face_net.display_original_image(face_imgs[0])
                # 在 label_30 上显示裁剪出的人脸
                self.ui.label_30.setPixmap(pixmap2.scaled(
                    self.ui.label_30.width(), self.ui.label_30.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))

                # 显示成功消息框
                # information() 显示信息提示对话框
                # self: 父窗口
                # "成功": 对话框标题
                # f"检测到 {len(face_imgs)} 张人脸\n已自动选择第一张人脸": 消息内容
                QMessageBox.information(
                    self,
                    "成功",
                    f"检测到 {len(face_imgs)} 张人脸\n已自动选择第一张人脸"
                )
            # 如果没有检测到人脸
            else:
                # 显示警告消息框
                # warning() 显示警告对话框
                QMessageBox.warning(self, "警告", "未检测到人脸，请更换图片！")

    # 定义开启摄像头进行人脸录入的方法
    def start_camera_enrollment(self):
        """开启摄像头进行人脸录入"""
        # 检查摄像头是否已经在运行
        # isOpened() 检查摄像头是否已打开
        if self.cap is not None and self.cap.isOpened():
            # 显示警告消息
            QMessageBox.warning(self, "警告", "摄像头已在运行中！")
            # 直接返回，不执行后续代码
            return

        # 尝试打开摄像头
        # VideoCapture(0) 打开默认摄像头（设备索引 0）
        self.cap = cv2.VideoCapture(0)
        # 检查摄像头是否成功打开
        if not self.cap.isOpened():
            # 显示严重错误消息框
            # critical() 显示严重错误对话框
            QMessageBox.critical(self, "错误", "无法打开摄像头！请检查设备连接。")
            # 返回
            return

        # 创建摄像头处理线程实例
        self.camera_thread = CameraThread()
        # 配置线程参数
        # cap: 摄像头对象
        # face_net: FaceNet 模型
        # 'enrollment': 录入模式
        # self.face_cache: 人脸缓存
        # 0.8: 识别阈值
        self.camera_thread.setup(self.cap, self.face_net, 'enrollment', self.face_cache, 0.8)
        # 连接线程的 frame_processed 信号到 update_frame 槽方法
        # 这样线程处理完的每一帧都会发送到主界面显示
        self.camera_thread.frame_processed.connect(self.update_frame)
        # 启动线程
        # start() 会调用 run() 方法在新线程中执行
        self.camera_thread.start()

    # 定义拍照进行人脸录入的方法
    def capture_face_enrollment(self):
        """拍照进行人脸录入"""
        # 检查摄像头是否已打开
        if self.cap is None or not self.cap.isOpened():
            # 显示警告消息
            QMessageBox.warning(self, "警告", "请先开启摄像头！")
            # 返回
            return
        # 检查是否有当前图像
        if self.current_image is None:
            # 显示警告消息
            QMessageBox.warning(self, "警告", "当前没有可用的摄像头画面！")
            # 返回
            return

        # 在当前画面中检测人脸
        # getfacepos() 方法检测并裁剪出所有人脸
        face_imgs = self.face_net.getfacepos(self.current_image)
        # 如果检测到人脸
        if face_imgs:
            # 保存检测到的人脸
            self.current_faces = face_imgs
            # 显示第一张检测到的人脸
            # display_original_image() 转换图片格式
            pixmap1 = self.face_net.display_original_image(face_imgs[0])
            # 在 label_30 上显示裁剪出的人脸
            self.ui.label_30.setPixmap(pixmap1.scaled(
                self.ui.label_30.width(), self.ui.label_30.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            # 显示成功消息
            QMessageBox.information(
                self,
                "成功",
                f"检测到 {len(face_imgs)} 张人脸\n已自动选择第一张人脸"
            )
        # 如果没有检测到人脸
        else:
            # 显示警告消息
            QMessageBox.warning(self, "警告", "当前画面未检测到人脸！")

    # 定义关闭摄像头的方法
    def stop_camera_enrollment(self):
        """关闭摄像头"""
        # 如果摄像头线程存在
        if self.camera_thread:
            # 调用线程的 stop() 方法停止运行
            self.camera_thread.stop()
            # wait() 等待线程完全结束
            # 这确保线程资源被正确释放
            self.camera_thread.wait()
            # 将线程引用设为 None
            self.camera_thread = None

        # 如果摄像头对象存在
        if self.cap is not None:
            # release() 方法释放摄像头资源
            self.cap.release()
            # 将摄像头对象设为 None
            self.cap = None

        # 清空显示
        # clear() 方法清除 label 上的内容
        self.ui.label_15.clear()
        self.ui.label_30.clear()

    # 定义更新摄像头帧的方法
    # 这是一个槽方法，接收线程发送的信号
    def update_frame(self, frame, recognition_result):
        """更新摄像头帧"""
        # 如果帧数据不为空
        if frame is not None:
            # 保存当前帧用于后续人脸捕获
            # copy() 创建深拷贝，避免数据被修改
            self.current_image = frame.copy()
            # 显示原始摄像头画面到中间 label
            # display_original_image() 转换图片格式
            pixmap1 = self.face_net.display_original_image(frame)
            # 在 label_15 上显示画面
            self.ui.label_15.setPixmap(pixmap1.scaled(
                self.ui.label_15.width(), self.ui.label_15.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    # 定义保存人脸信息到数据库的方法
    def save_face_info(self):
        """保存人脸信息到数据库"""
        # 获取输入框中的姓名
        # text() 获取文本内容
        # strip() 去除首尾空格
        name = self.ui.lineEdit_13.text().strip()
        # 获取输入框中的年龄
        age = self.ui.lineEdit_14.text().strip()
        # 获取输入框中的学号
        student_id = self.ui.lineEdit_15.text().strip()
        # 初始化性别变量为空字符串
        gender = ""
        # 检查单选按钮的选择状态
        # isChecked() 检查是否被选中
        if self.ui.radioButton_5.isChecked():  # 男
            # 如果"男"单选按钮被选中
            gender = "男"
        elif self.ui.radioButton_6.isChecked():  # 女
            # 如果"女"单选按钮被选中
            gender = "女"
        # 验证输入是否完整
        # 如果任意字段为空
        if not name or not age or not student_id or not gender:
            # 显示警告消息
            QMessageBox.warning(self, "警告", "请填写完整信息！")
            # 返回
            return
        # 尝试将年龄转换为整数
        try:
            # int() 将字符串转换为整数
            age = int(age)
        # 如果转换失败（抛出 ValueError 异常）
        except ValueError:
            # 显示警告消息
            QMessageBox.warning(self, "警告", "年龄必须是数字！")
            # 返回
            return
        # 检查是否检测到人脸
        if not self.current_faces:
            # 显示警告消息
            QMessageBox.warning(self, "警告", "没有检测到人脸！")
            # 返回
            return
        # 计算人脸特征向量
        try:
            # 获取第一张人脸图像
            face_img = self.current_faces[0]
            # 使用 FaceNet 提取特征向量
            embedding = self.face_net.facenet(face_img)
        # 如果特征提取失败
        except Exception as e:
            # 显示严重错误消息
            QMessageBox.critical(self, "错误", f"人脸特征提取失败：{str(e)}")
            # 返回
            return
        # 将图像转换为字节流存储
        # imencode() 将图像编码为 JPEG 格式
        # _: 返回的第一个值是编码成功的标志（这里不需要）
        # img_encoded: 编码后的图像数据（numpy 数组）
        _, img_encoded = cv2.imencode('.jpg', face_img)
        # tobytes() 将 numpy 数组转换为字节流
        photo_bytes = img_encoded.tobytes()
        # 获取当前时间
        # datetime.now() 获取当前日期和时间
        # strftime() 格式化为字符串
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 定义插入数据库的 SQL 语句
        # INSERT INTO: 插入数据
        # ?: 占位符，防止 SQL 注入
        sql = """
                INSERT INTO student_info (姓名，年龄，性别，学号，录入时间，照片)
                VALUES (?, ?, ?, ?, ?, ?)
                """
        # 执行 SQL 语句
        # operation_sql() 方法带参数执行
        # 参数列表按顺序填充占位符
        result = self.db.operation_sql(sql, [name, age, gender, student_id, current_time, photo_bytes])
        # 如果插入成功
        if result:
            # 显示成功消息
            QMessageBox.information(self, "成功", "人脸信息保存成功！")
            # 清空输入框
            self.ui.lineEdit_13.clear()
            self.ui.lineEdit_14.clear()
            self.ui.lineEdit_15.clear()
            # 取消单选按钮的选择状态
            self.ui.radioButton_5.setChecked(False)
            self.ui.radioButton_6.setChecked(False)
            # 清除人脸显示
            self.ui.label_30.clear()
            # 清空当前人脸列表
            self.current_faces = []

            # 刷新缓存
            # 重新从数据库加载人脸特征
            self._refresh_cache()
        # 如果插入失败
        else:
            # 显示严重错误消息
            QMessageBox.critical(self, "错误", "保存失败")

    # 定义上传图片进行人脸识别的方法
    def upload_image_for_recognition(self):
        """上传图片进行人脸识别"""
        # 打开文件选择对话框选择识别图片
        file_path, _ = QFileDialog.getOpenFileName(self, "选择识别图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        # 如果用户选择了文件
        if file_path:
            # 在输入框中显示文件路径
            self.ui.lineEdit_11.setText(file_path)
            # 读取图片
            original_img = cv2.imread(file_path)
            # 如果图片读取成功
            if original_img is not None:
                # 显示原始图像
                pixmap1 = self.face_net.display_original_image(original_img)
                # 在 label_24 上显示图片
                self.ui.label_24.setPixmap(pixmap1.scaled(
                    self.ui.label_24.width(), self.ui.label_24.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                # 检测人脸
                face_imgs = self.face_net.getfacepos(file_path)
                # 如果检测到人脸
                if face_imgs:
                    # 初始化识别结果列表
                    recognition_results = []
                    # 遍历所有检测到的人脸
                    for i, face_img in enumerate(face_imgs):
                        # 提取人脸特征
                        try:
                            # 使用 FaceNet 提取特征向量
                            embedding = self.face_net.facenet(face_img)
                            # 查找匹配的人脸
                            # find_matches_from_cache() 在缓存中查找最相似的人脸
                            match_result = self.find_matches_from_cache(embedding)
                            # 如果找到匹配
                            if match_result:
                                # 解包匹配结果
                                # name: 姓名
                                # confidence: 置信度
                                name, confidence = match_result
                                # 格式化识别结果
                                recognition_results.append(f"人脸{i+1}: {name} (置信度：{confidence:.2f})")
                            # 如果没有找到匹配
                            else:
                                # 标记为未知人员
                                recognition_results.append(f"人脸{i+1}: 未知人员 (置信度：0.00)")
                        # 如果特征提取失败
                        except Exception as e:
                            # 记录错误信息
                            recognition_results.append(f"人脸{i+1}: 特征提取失败 ({str(e)})")
                    # 合并所有识别结果
                    # join() 用换行符连接列表中的所有字符串
                    result_text = "\n".join(recognition_results)
                    # 在文本编辑器中显示识别结果
                    self.ui.plainTextEdit_2.setPlainText(result_text)
                # 如果没有检测到人脸
                else:
                    # 显示警告消息
                    QMessageBox.warning(self, "警告", "未检测到人脸！")
                    # 在文本编辑器中显示提示
                    self.ui.plainTextEdit_2.setPlainText("未检测到人脸！")
            # 如果图片读取失败
            else:
                # 显示严重错误消息
                QMessageBox.critical(self, "错误", "无法读取图片文件！")
                # 在文本编辑器中显示错误
                self.ui.plainTextEdit_2.setPlainText("无法读取图片文件！")

    # 定义从缓存中查找最相似人脸的方法
    def find_matches_from_cache(self, query_embedding, threshold=0.8):
        """从缓存中查找最相似的人脸"""
        # 初始化最小距离为正无穷大
        min_distance = float('inf')
        # 初始化最佳匹配为 None
        best_match = None

        # 遍历人脸缓存中的每个人脸数据
        for person_data in self.face_cache:
            # 获取数据库中存储的人脸特征向量
            db_embedding = person_data['embedding']
            # 计算查询特征向量与数据库特征向量之间的欧氏距离
            distance = np.linalg.norm(query_embedding - db_embedding)

            # 如果当前距离小于已知最小距离，并且小于阈值
            if distance < min_distance and distance < threshold:
                # 更新最小距离
                min_distance = distance
                # 计算置信度
                confidence = 1 - (distance / threshold)
                # 保存最佳匹配结果（只保存姓名和置信度）
                best_match = (person_data['info']['姓名'], confidence)

        # 返回最佳匹配结果
        return best_match

    # 定义开启摄像头进行人脸识别的方法
    def start_camera_recognition(self):
        """开启摄像头进行人脸识别"""
        # 检查摄像头是否已经在运行
        if self.cap is not None and self.cap.isOpened():
            # 显示警告消息
            QMessageBox.warning(self, "警告", "摄像头已在运行中！")
            # 返回
            return

        # 尝试打开摄像头
        self.cap = cv2.VideoCapture(0)
        # 检查摄像头是否成功打开
        if not self.cap.isOpened():
            # 显示严重错误消息
            QMessageBox.critical(self, "错误", "无法打开摄像头！请检查设备连接。")
            # 返回
            return

        # 创建摄像头处理线程实例
        self.camera_thread = CameraThread()
        # 配置线程参数
        # 'recognition': 识别模式
        self.camera_thread.setup(self.cap, self.face_net, 'recognition', self.face_cache, 0.8)
        # 连接线程的 frame_processed 信号到 update_recognition_frame 槽方法
        self.camera_thread.frame_processed.connect(self.update_recognition_frame)
        # 启动线程
        self.camera_thread.start()

    # 定义更新识别帧的方法
    def update_recognition_frame(self, frame, recognition_result):
        """更新识别帧"""
        # 如果帧数据不为空
        if frame is not None:
            # 显示处理后的帧
            pixmap1 = self.face_net.display_original_image(frame)
            # 在 label_24 上显示画面
            self.ui.label_24.setPixmap(pixmap1.scaled(
                self.ui.label_24.width(), self.ui.label_24.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            # 更新识别结果文本
            # 在 plainTextEdit_2 中显示识别结果
            self.ui.plainTextEdit_2.setPlainText(recognition_result)

    # 定义关闭摄像头识别的方法
    def stop_camera_recognition(self):
        """关闭摄像头识别"""
        # 如果摄像头线程存在
        if self.camera_thread:
            # 停止线程
            self.camera_thread.stop()
            # 等待线程结束
            self.camera_thread.wait()
            # 清空线程引用
            self.camera_thread = None

        # 如果摄像头对象存在
        if self.cap is not None:
            # 释放摄像头资源
            self.cap.release()
            # 清空摄像头引用
            self.cap = None
        # 清空显示
        self.ui.label_24.clear()
        self.ui.plainTextEdit_2.clear()

    # 定义搜索数据库的方法
    def search_database(self):
        """搜索数据库"""
        # 简单实现：根据学号或姓名搜索
        # 获取搜索关键词输入框的内容
        keyword = self.ui.lineEdit_12.text().strip()
        # 如果关键词为空
        if not keyword:
            # 显示警告消息
            QMessageBox.warning(self, "警告", "请输入查询关键词！")
            # 返回
            return

        # 模糊查询 姓名 或 学号
        # LIKE: SQL 模糊匹配操作符
        # %: 通配符，表示任意字符
        # WHERE 姓名 LIKE ? OR 学号 LIKE ?: 条件为姓名或学号包含关键词
        sql = "SELECT ID, 姓名，年龄，性别，学号，录入时间 FROM student_info WHERE 姓名 LIKE ? OR 学号 LIKE ?"
        # 构造参数，前后加%实现模糊匹配
        param = f"%{keyword}%"
        # 执行 SQL 查询
        result = self.db.operation_sql(sql, [param, param])

        # 清空并重新填充表格
        # setRowCount(0) 清空所有行
        self.ui.tableView_2.setRowCount(0)
        # 定义表头列表
        headers = ["ID", "姓名", "年龄", "性别", "学号", "录入时间"]
        # 设置表格列数
        self.ui.tableView_2.setColumnCount(len(headers))
        # 设置水平表头标签
        self.ui.tableView_2.setHorizontalHeaderLabels(headers)
        # 获取水平表头对象
        header = self.ui.tableView_2.horizontalHeader()
        # 设置列宽模式为拉伸模式（自动适应宽度）
        # Stretch: 列宽自动伸展以填充整个表格宽度
        header.setSectionResizeMode(QHeaderView.Stretch)

        # 如果有查询结果
        if result:
            # 设置表格行数
            self.ui.tableView_2.setRowCount(len(result))
            # 遍历每一行数据
            for row_idx, row_data in enumerate(result):
                # 遍历每一列数据
                for col_idx, data in enumerate(row_data):
                    # 如果是字节类型数据
                    if isinstance(data, bytes):
                        # 显示为"BLOB 数据"
                        data = "BLOB 数据"
                    # 创建表格项
                    # str(data) 将数据转换为字符串
                    # if data is not None else "" 如果数据为 None 则显示空字符串
                    item = QTableWidgetItem(str(data) if data is not None else "")
                    # 设置文本对齐方式为居中对齐
                    item.setTextAlignment(Qt.AlignCenter)
                    # 将表格项设置到表格的指定位置
                    self.ui.tableView_2.setItem(row_idx, col_idx, item)

    # 定义加载表格数据的方法
    def load_table_data(self):
        """加载表格数据"""
        # 清空现有表格
        # setRowCount(0) 删除所有行
        self.ui.tableView_2.setRowCount(0)

        # 设置表头
        headers = ["ID", "姓名", "年龄", "性别", "学号", "录入时间"]
        # 设置表格列数
        self.ui.tableView_2.setColumnCount(len(headers))
        # 设置水平表头标签
        self.ui.tableView_2.setHorizontalHeaderLabels(headers)

        # 设置列宽自适应
        header = self.ui.tableView_2.horizontalHeader()
        # Stretch 模式让列宽自动适应
        header.setSectionResizeMode(QHeaderView.Stretch)

        # 查询数据库
        # SELECT 语句获取所有记录（不包含照片字段）
        sql = "SELECT ID, 姓名，年龄，性别，学号，录入时间 FROM student_info"
        # 执行查询
        result = self.db.operation_sql(sql)

        # 如果有查询结果
        if result:
            # 设置表格行数
            self.ui.tableView_2.setRowCount(len(result))
            # 遍历每一行数据
            for row_idx, row_data in enumerate(result):
                # 遍历每一列数据
                for col_idx, data in enumerate(row_data):
                    # 处理可能存在的字节数据或其他类型，确保转为字符串
                    if isinstance(data, bytes):
                        # 照片中不直接显示二进制
                        data = "BLOB 数据"
                    # 创建表格项
                    item = QTableWidgetItem(str(data) if data is not None else "")
                    # 设置文本居中对齐
                    item.setTextAlignment(Qt.AlignCenter)
                    # 将表格项设置到表格中
                    self.ui.tableView_2.setItem(row_idx, col_idx, item)

    # 定义删除记录的方法
    def delete_record(self):
        """删除记录"""
        # 获取当前选中的行
        # selectionModel() 获取选择模型
        # selectedRows() 获取所有选中的行索引
        selected_rows = self.ui.tableView_2.selectionModel().selectedRows()
        # 如果没有选中任何行
        if not selected_rows:
            # 显示警告消息
            QMessageBox.warning(self, "警告", "请先选择要删除的记录！")
            # 返回
            return

        # 确认删除
        # question() 显示询问对话框
        # '确认删除': 对话框标题
        # '确定要删除选中的记录吗？': 询问内容
        # QMessageBox.Yes | QMessageBox.No: 显示"是"和"否"按钮
        # QMessageBox.No: 默认按钮
        reply = QMessageBox.question(self, '确认删除', '确定要删除选中的记录吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # 如果用户选择"是"
        if reply == QMessageBox.Yes:
            # 从后往前删除，避免索引变化
            # sorted() 对选中的行排序
            # key=lambda x: x.row() 按行号排序
            # reverse=True 逆序排列（从大到小）
            for index in sorted(selected_rows, key=lambda x: x.row(), reverse=True):
                # 获取行号
                row = index.row()
                # 获取该行的 ID (假设第一列是 ID)
                # item(row, 0) 获取第 row 行第 0 列的表格项
                id_item = self.ui.tableView_2.item(row, 0)
                # 如果 ID 项存在
                if id_item:
                    # 获取 ID 值
                    record_id = id_item.text()
                    # 定义删除记录的 SQL 语句
                    # DELETE FROM: 删除记录
                    # WHERE ID = ?: 条件是 ID 等于指定值
                    sql = "DELETE FROM student_info WHERE ID = ?"
                    # 执行删除操作
                    result = self.db.operation_sql(sql, [record_id])
                    # 如果删除失败
                    if not result:
                        # 显示错误消息
                        QMessageBox.critical(self, "错误", f"删除记录 ID={record_id} 失败")

            # 删除后刷新表格
            self.load_table_data()
            # 刷新缓存
            # 更新人脸特征缓存
            self._refresh_cache()
            # 显示成功消息
            QMessageBox.information(self, "成功", "记录删除成功！")

    # 定义刷新表格的方法
    def refresh_table(self):
        """刷新表格"""
        # 调用加载表格数据方法
        # 重新从数据库加载所有数据并显示
        self.load_table_data()

    # 定义窗口关闭事件处理方法
    # 当用户关闭窗口时会自动调用此方法
    def closeEvent(self, event):
        """关闭事件处理"""
        # 如果摄像头线程存在
        if self.camera_thread:
            # 停止线程
            self.camera_thread.stop()
            # 等待线程结束
            self.camera_thread.wait()

        # 如果摄像头对象存在
        if self.cap is not None:
            # 释放摄像头资源
            self.cap.release()
        # 接受关闭事件
        # 如果不调用 accept()，窗口可能不会关闭
        event.accept()


# Python 脚本的主入口判断
# 只有直接运行此文件时才会执行以下代码
if __name__ == '__main__':
    # 创建 QApplication 实例
    # sys.argv 是命令行参数列表
    # QApplication 管理 GUI 应用程序的控制流和主要设置
    app = QApplication(sys.argv)
    # 创建主窗口实例
    window = MainWindow()
    # 显示窗口
    # show() 方法使窗口可见
    window.show()
    # 启动应用程序的主循环
    # exec_() 方法开始事件循环
    # sys.exit() 确保程序正常退出
    sys.exit(app.exec_())
