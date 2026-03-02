
import cv2
import queue
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtSql import QSqlQuery, QSqlDatabase
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO


class cameraThread(QThread):
    send_result_img = pyqtSignal(QImage)
    send_original_frame = pyqtSignal(QImage)
    send_face_feature_box = pyqtSignal(list, list, QImage)

    def __init__(self, yolov8_model_path):
        super().__init__()
        self.running = True
        # 加载 YOLO 模型，准备预测
        self.yolov8_model = YOLO(yolov8_model_path)
        # 加载 FaceNet 模型，用于提取人脸特征
        self.facenet_model = InceptionResnetV1(pretrained='casia-webface').eval().to('cpu')
        # 用于存储多个人脸的边界框坐标列表
        self.face_boxes = []
        # 用于存储多个人脸的特征列表
        self.face_features = []

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while self.running and self.cap.isOpened():
            # 注意 frame 是 numpy 数组， 使用信号传递参数的时候要注意信号参数的定义
            success, self.frame = self.cap.read()
            # 如果读取成功
            if success:
                original_frame = self.frame.copy()
                # 使用 YOLOv8 模型对当前帧进行预测，conf是置信度阈值
                results = self.yolov8_model.predict(self.frame, conf=0.7)
                # 获取检测结果中的边界框信息
                boxes = results[0].boxes
                # 清空两个列表
                self.face_boxes.clear()
                self.face_features.clear()
                # 遍历检测到的人脸框列表
                for box in boxes:
                    # 解包人脸框的坐标信息，xyxy是一个包含左上角和右下角坐标的列表
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # 提取类别标签，这里类别 0 代表人脸
                    cls = box.cls[0].tolist()
                    # 如果类别标签为0（即为人脸）
                    if cls == 0:
                        # 在原始帧上画一个红色的框来标记人脸
                        cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        # 从原始帧中裁剪出人脸图像区域
                        face_img = self.frame[int(y1):int(y2), int(x1):int(x2)]
                        # 创建一个列表包含边界框坐标, 如果一张图上有两张人脸，结果[[x1, y1], [x2, y2]]
                        self.face_boxes.append([x1, y1])
                        # 检查裁剪出的人脸图像是否非空
                        if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                            # 对裁剪出的人脸图像进行预处理
                            face_tensor = self.preprocess_face_img(face_img)
                            # 从预处理后的人脸图像中提取特征
                            face_feature = self.extract_face_feature(face_tensor)
                            self.face_features.append(face_feature)

                # 通过信号传输画红框的图像
                # BGR -> RGB
                rgb_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                # 转换成 QImage
                rgb_img = QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], QImage.Format_RGB888)
                # 使用信号传递 frame
                self.send_result_img.emit(rgb_img)

                # 处理原图并传输
                h, w, ch = original_frame.shape
                original_frame_bytes_per_line = ch * w
                original_frame = QImage(original_frame.data, w, h, original_frame_bytes_per_line, QImage.Format_BGR888)
                self.send_original_frame.emit(original_frame)

                self.send_face_feature_box.emit(self.face_boxes, self.face_features, rgb_img)

        # 线程结束，释放摄像头资源
        self.cap.release()

    def preprocess_face_img(self, face_img):
        """
        对人脸图像进行预处理
        :param face_img: 需要处理的图像
        :return:返回处理图像后的 PyTorch 张量
        """
        # 将BGR图像转换为RGB图像
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # 定义 FaceNet 模型所需的输入图像尺寸
        required_size = (160, 160)
        # 使用OpenCV调整图像尺寸到模型期望的尺寸
        face_img = cv2.resize(face_img, required_size)
        # 将图像数据转换为 PyTorch 张量
        # permute(2, 0, 1)将图像的维度从HWC (高度、宽度、通道) 转换为CHW (通道、高度、宽度)
        # float()将张量数据类型转换为浮点数
        # 将图像转换为Tensor格式，并归一化
        # unsqueeze(0)在张量前面添加一个批次维度，因为模型期望输入是4维的 [B, C, H, W]
        face_tensor = (torch.tensor(face_img).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
        # 返回图像归一化后的 PyTorch 张量
        return face_tensor

    def extract_face_feature(self, face_tensor):
        """
        提取人脸特征
        :param face_tensor: 图像预处理后的 PyTorch 张量
        :return: 返回 face_embedding 人脸特征
        """
        # 使用 torch.no_grad() 上下文管理器，表示告诉 PyTorch 不需要计算梯度
        # 这通常用于推理阶段，可以减少内存消耗和提高速度
        with torch.no_grad():
            # 将处理后的图像张量传递给 FaceNet 模型以提取特征
            face_embedding = self.facenet_model(face_tensor)
            # 对提取的人脸特征进行L2归一化
            # 计算特征向量的L2范数
            l2_norm = torch.norm(face_embedding, p=2, dim=1, keepdim=True)
            # 将特征向量除以其L2范数进行归一化
            face_embedding_normalized = face_embedding.div(l2_norm)
        # 将得到的特征张量移动到CPU上，并转换为 NumPy 数组
        # 这一步是为了后续处理，如保存特征或进行其他非 PyTorch 操作
        return face_embedding_normalized.cpu().numpy()

    def stop(self):
        # 线程停止标志位
        self.running = False
        self.wait()  # 等待线程结束


class MySqlThread(QThread):
    INSERT_NEW = 1  # 定义常量，表示插入数据
    SELECT_BY_KEYWORD = 2  # 定义常量，表示查询的命令
    SELECT_FACE_FEATURE = 3  # 定义常量，表示查询人脸特征的命令
    UPDATE_SORT = 4  # # 定义常量，表示重新排序

    query_result_signal = pyqtSignal(list)  # 发送查询结果的信号
    signal_sql_data = pyqtSignal(dict)  # 发送 数据库的人脸特征这一行的数据的信号
    finished_signal = pyqtSignal()  # 更新 界面 model 的信号

    def __init__(self):
        """
        初始化数据库，并连接
        """
        super().__init__()
        # 初始化队列，用于存储待处理的 SQL 命令
        self.q_deal_sql_cmd = queue.Queue(5)
        # 停止线程的标志位
        self.running = True
        try:
            # 创建一个 SQLite 数据库连接
            self.face_database = QSqlDatabase.addDatabase("QSQLITE")
            # 设置数据库文件名，这将创建或连接到一个名为 "mydatabase.db" 的SQLite数据库文件
            self.face_database.setDatabaseName("./static/sqlite/FaceDatabase.db")
            # 尝试打开数据库连接，如果数据库文件不存在，它将被创建
            self.face_database.open()
            # 尝试打开数据库连接，如果数据库文件不存在，它将被创建
            if not self.face_database.open():
                raise Exception("无法打开数据库连接")
            print('已连接到数据库')
            # 检查并创建 facedatabase 表（如果它还不存在）
            self.check_and_create_table()
        except Exception as e:
            # 打印错误信息
            print(f'无法连接到数据库！error:{e}')
        # 创建一个 QSqlQuery 对象，用于执行 SQL 语句
        self.query = QSqlQuery()

    def check_and_create_table(self):
        """
        检查 facedatabase 表是否存在，如果不存在则创建
        """
        # 创建一个 QSqlQuery 对象，用于执行SQL语句
        check_query = QSqlQuery(self.face_database)

        # 检查表是否存在
        check_query.exec_("SELECT name FROM sqlite_master WHERE type='table' AND name='facedatabase';")
        if not check_query.next():
            print('表 "facedatabase" 不存在，正在创建...')
            # 表不存在，创建表
            create_table_query = QSqlQuery(self.face_database)
            sql_statement = """
                    CREATE TABLE facedatabase (
                        序号 INTEGER PRIMARY KEY,  -- 主键，自动递增
                        姓名 TEXT NOT NULL,  -- 不能为空
                        性别 TEXT,
                        年龄 INTEGER,
                        学号 INTEGER,
                        录入时间 TEXT,
                        照片 BLOB,  -- 照片字段应该是 BLOB 类型
                        人脸特征 BLOB
                    );
                    """
            if not create_table_query.exec_(sql_statement):
                print(f'创建表格“facedatabase”失败: {create_table_query.lastError().text()}')
            else:
                print('表 "facedatabase" 创建成功。')
        else:
            print('表 "facedatabase" 已存在。')

    def run(self):
        # 开启循环，等待执行命令
        while self.running:
            # 从队列中获取SQL命令和数据
            q_data = self.q_deal_sql_cmd.get()  # 格式：{"cmd": "INSERT", "content": "xxxx"}
            cmd = q_data["cmd"]
            # 如果命令是插入命令
            if cmd == self.INSERT_NEW:
                # 获取插入的数据
                content = q_data["content"]
                self.query.prepare("INSERT INTO "
                                   "facedatabase (姓名, 性别, 年龄, 学号, 录入时间, 照片, 人脸特征) "
                                   "VALUES (:name, :sex, :age, :studentID, :time, :img, :facefeature)")
                self.query.exec_()
                # 遍历内容字典中的每个键值对
                for key in content.keys():
                    # 将每个键值对绑定到对应的命名占位符上
                    # 这样可以确保数据被正确地插入到数据库中，并且避免了SQL注入的风险
                    # print(f'key：:{key}, value：{content[key]}')
                    self.query.bindValue(f":{key}", content[key])
                # 执行SQL插入语句并检查结果
                if not self.query.exec_():
                    print(f'Error executing query: {self.query.lastError().text()}')
                    print(f'SQL错误码: {self.query.lastError().nativeErrorCode()}')
                    print(f'SQL错误驱动程序文本: {self.query.lastError().driverText()}')
                    print(f'SQL错误数据库文本: {self.query.lastError().databaseText()}')
                else:
                    print('数据已成功保存。')
            # 判断接收到的命令是否为  查询（依据姓名查询）
            elif cmd == self.SELECT_BY_KEYWORD:
                content = q_data["content"]
                # 只支持查询姓名
                if "name" in content:
                    name = content["name"].strip()
                    # 构建 SQL 查询语句
                    sql = "SELECT * FROM facedatabase WHERE 姓名 LIKE :keyword LIMIT 1"
                    # 准备并执行查询
                    self.query.prepare(sql)
                    self.query.bindValue(":keyword", f"%{name}%")
                    if not self.query.exec_():
                        print(f'查询错误: {self.query.lastError().text()}')
                        self.query_result_signal.emit([])  # 发送空列表表示查询失败
                    else:
                        result = []
                        if self.query.first():
                            record = self.query.record()
                            row_data = []
                            for i in range(record.count()):
                                record.fieldName(i)
                                value = self.query.value(i)
                                row_data.append(value)
                            result.append(row_data)
                        self.query_result_signal.emit(result)  # 发送查询结果
            # 判断接收到的命令是否为查询人脸特征（获取数据库数据便于匹配后显示）
            elif cmd == self.SELECT_FACE_FEATURE:
                sql = "SELECT * FROM facedatabase"
                self.query.prepare(sql)
                self.query.exec_()
                # 创建一个空列表，放置信息
                data = []
                while self.query.next():
                    db_name = self.query.value(1)
                    db_sex = self.query.value(2)
                    db_age = self.query.value(3)
                    db_studentID = self.query.value(4)
                    db_facefeature = self.query.value(7)
                    data.append((db_name, db_sex, db_age, db_studentID, db_facefeature))
                self.signal_sql_data.emit({self.SELECT_FACE_FEATURE: data})
            elif cmd == self.UPDATE_SORT:
                try:
                    # 开启数据库事务
                    self.face_database.transaction()
                    # 创建临时表备份原表数据
                    sql_temp_table = "CREATE TABLE temp_table AS SELECT * FROM facedatabase"
                    self.query.prepare(sql_temp_table)
                    if not self.query.exec_():
                        raise Exception(f"创建临时表失败: {self.query.lastError().text()}")

                    # 清空原表数据
                    sql_clear_original = "DELETE FROM facedatabase"
                    self.query.prepare(sql_clear_original)
                    if not self.query.exec_():
                        raise Exception(f"清空原表失败: {self.query.lastError().text()}")

                    # 重新插入数据到原表，让数据库自动处理序号
                    sql_insert_data = """
                                        INSERT INTO facedatabase (姓名, 性别, 年龄, 学号, 录入时间, 照片, 人脸特征)
                                        SELECT 姓名, 性别, 年龄, 学号, 录入时间, 照片, 人脸特征
                                        FROM temp_table
                                        """
                    self.query.prepare(sql_insert_data)
                    if not self.query.exec_():
                        raise Exception(f"从临时表插入数据失败: {self.query.lastError().text()}")

                    # 提交事务，使更新操作持久化到数据库
                    self.face_database.commit()
                except Exception as e:
                    print(f"执行排序刷新操作出现错误: {e}")
                    print(f"查询状态: {self.query.lastQuery()}, 绑定值: {self.query.boundValues()}")
                    print(f"数据库连接有效: {self.face_database.isValid()}, 数据库已打开: {self.face_database.isOpen()}")
                    # 出现异常回滚事务，恢复到之前的数据库状态
                    self.face_database.rollback()
                finally:
                    # 检查临时表是否存在
                    self.query.exec_("SELECT * FROM temp_table LIMIT 1")
                    if self.query.next():  # 如果表存在
                        sql_drop_temp_table = "DROP TABLE IF EXISTS temp_table"
                        if not self.query.exec_(sql_drop_temp_table):
                            print(f"在 finally 块中删除临时表失败: {self.query.lastError().text()}")
                            print(f"查询状态: {self.query.lastQuery()}, 绑定值: {self.query.boundValues()}")
                    else:
                        print("临时表不存在，无需删除。")
                    self.finished_signal.emit()

    def stop(self):
        self.running = False
        # 断开数据库的连接
        self.face_database.close()
        # 等待线程结束
        self.wait()
