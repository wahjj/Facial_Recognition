# 导入 torch 库，用于深度学习模型推理
import torch
# 导入 pathlib 的 Path 类，用于处理文件和目录路径
from pathlib import Path
# 导入 OpenCV 库，用于图像处理和人脸检测
import cv2
# 导入 numpy 库，用于数值计算和数组操作
import numpy as np
# 从 PyQt5 的 QtGui 模块导入图像显示相关类
# QImage 用于加载和操作图像
# QPixmap 用于在 PyQt 界面中显示图像
from PyQt5.QtGui import QImage, QPixmap

# 从 ultralytics 库导入 YOLO 模型
# YOLO 是 You Only Look Once 的缩写，用于实时目标检测
from ultralytics import YOLO
# 从 facenet_pytorch 库导入 InceptionResnetV1 模型
# 这是 FaceNet 的一种实现，用于人脸特征提取
from facenet_pytorch import InceptionResnetV1


# 定义 FaceNet 类，封装人脸识别的所有功能
class FaceNet:
    """
    FaceNet 人脸识别类
    集成 YOLO 人脸检测和 FaceNet 特征提取功能
    """
    # 类的初始化方法，在创建 FaceNet 实例时自动执行
    def __init__(self, yolo_model_path):
        """
        初始化 FaceNet 类
        :param yolo_model_path: YOLO 模型权重文件路径
        """
        # 加载预训练的 FaceNet 模型（使用 CASIA-WebFace 数据集预训练）
        # InceptionResnetV1 是一种结合了 Inception 和 ResNet 架构的深度神经网络
        # pretrained='casia-webface' 表示使用在 CASIA-WebFace 数据集上预训练的权重
        # .eval() 将模型设置为评估模式（区别于训练模式）
        # .to('cpu') 将模型移动到 CPU 设备上进行推理
        self.facenet_model = InceptionResnetV1(pretrained='casia-webface').eval().to('cpu')
        # 加载 YOLO 目标检测模型
        # YOLO() 加载指定路径的模型权重文件
        self.yolo_model = YOLO(yolo_model_path)

    # 定义人脸图像预处理方法
    def preprocess_face_img(self, face_img):
        """
        预处理人脸图像，转换为模型输入格式
        :param face_img: OpenCV 读取的人脸图像（BGR 格式）
        :return: 预处理后的张量
        """
        # BGR 转 RGB（OpenCV 默认是 BGR，需要转为 RGB）
        # cv2.cvtColor() 进行颜色空间转换
        # cv2.COLOR_BGR2RGB 指定从 BGR 转换到 RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # 缩放到 FaceNet 要求的尺寸（160x160）
        # cv2.resize() 调整图像大小
        # (160, 160) 指定目标尺寸（宽度，高度）
        face_img = cv2.resize(face_img, (160, 160))
        # 转换为张量并归一化：像素值从 [0, 255] 缩放到 [0, 1]
        # torch.tensor() 将 numpy 数组转换为 PyTorch 张量
        # .permute(2, 0, 1) 将 HWC 格式（高×宽×通道）转为 CHW 格式（通道×高×宽）
        # / 255.0 将像素值归一化到 0-1 范围
        # .unsqueeze(0) 在维度 0 增加一个维度，变为 (1, 3, 160, 160)，满足批量输入要求
        face_tensor = (torch.tensor(face_img).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
        # 返回预处理后的张量
        return face_tensor

    # 使用 FaceNet 提取人脸特征向量的方法
    def facenet(self, face_img):
        """
        使用 FaceNet 提取人脸特征向量
        :param face_img: OpenCV 读取的人脸图像
        :return: L2 归一化后的人脸特征向量
        """
        # 调用预处理方法处理输入图像
        face_tensor = self.preprocess_face_img(face_img=face_img)
        # 禁用梯度计算，减少内存消耗
        # torch.no_grad() 上下文管理器，告诉 PyTorch 不需要计算梯度
        with torch.no_grad():
            # 通过 FaceNet 模型提取特征
            # 将预处理后的张量输入模型，得到输出特征向量
            face_embedding = self.facenet_model(face_tensor)
            # L2 归一化：将特征向量归一化为单位向量，便于后续相似度计算
            # torch.norm() 计算范数
            # p=2 表示计算 L2 范数（欧几里得范数）
            # dim=1 沿着维度 1（特征维度）计算
            # keepdim=True 保持输出维度与原维度一致
            l2_norm = torch.norm(face_embedding, p=2, dim=1, keepdim=True)
            # 将特征向量除以 L2 范数，得到单位向量
            # .div() 是逐元素除法
            face_embedding_normalized = face_embedding.div(l2_norm)
        # 转换回 numpy 数组并移到 CPU
        # .cpu() 确保张量在 CPU 上
        # .numpy() 将 PyTorch 张量转换为 numpy 数组
        return face_embedding_normalized.cpu().numpy()

    # 检测图像中的人脸并裁剪出人脸区域的方法
    def getfacepos(self, img):
        """
        检测图像中的人脸并裁剪出人脸区域
        :param img: 可以是图片路径（str）、目录路径（str）或摄像头帧（numpy.ndarray）
        :return: 裁剪出的人脸图像列表
        """
        # 初始化一个空列表，用于存储裁剪出的人脸图像
        face_imgs = []

        # 判断输入是图片路径还是 numpy 数组（摄像头帧）
        # isinstance() 检查对象类型
        if isinstance(img, str):
            # 如果是字符串，当作文件路径处理
            # Path() 创建路径对象
            img_path = Path(img)
            # 如果是目录：批量处理目录下所有 jpg 图片
            # is_dir() 检查路径是否为目录
            if img_path.is_dir():
                # glob('*.jpg') 获取目录下所有.jpg 文件
                # 返回一个生成器，可以迭代所有匹配的文件
                for image_path in img_path.glob('*.jpg'):
                    # cv2.imread() 读取图片
                    # str(image_path) 将 Path 对象转为字符串
                    frame = cv2.imread(str(image_path))
                    # 使用 YOLO 检测人脸
                    # predict() 进行目标检测
                    # conf=0.7 设置置信度阈值为 0.7，只保留置信度高于 0.7 的检测结果
                    # verbose=False 关闭详细输出
                    results = self.yolo_model.predict(frame, conf=0.7, verbose=False)
                    # 获取检测框信息
                    # results[0] 获取第一张图片的检测结果（支持批量输入）
                    # boxes 属性包含所有检测框的信息
                    boxes = results[0].boxes
                    # 遍历所有检测框
                    for box in boxes:
                        # 获取检测框的坐标
                        # xyxy[0] 获取第一个框的坐标（左上角 x,y 和右下角 x,y）
                        # .tolist() 将张量转换为列表
                        # map(int, ...) 将所有坐标转换为整数
                        # int() 转换浮点数为整数
                        x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                        # 根据检测框裁剪人脸区域
                        # frame[y0:y1, x0:x1] 使用 numpy 切片操作裁剪图像
                        face_img = frame[y0:y1, x0:x1]
                        # 将裁剪出的人脸图像添加到列表中
                        face_imgs.append(face_img)
            # 如果是单张图片
            # is_file() 检查路径是否为文件
            elif img_path.is_file():
                # 读取单张图片
                frame = cv2.imread(str(img_path))
                # 使用 YOLO 检测人脸
                results = self.yolo_model.predict(frame, conf=0.7, verbose=False)
                # 获取检测框信息
                boxes = results[0].boxes
                # 遍历所有检测框
                for box in boxes:
                    # 获取检测框坐标
                    x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                    # 裁剪人脸区域
                    face_img = frame[y0:y1, x0:x1]
                    # 添加到列表
                    face_imgs.append(face_img)
        # 如果是 numpy 数组（摄像头帧）
        # ndarray 是 numpy 的多维数组类型
        elif isinstance(img, np.ndarray):
            # 直接使用输入的数组作为图像帧
            frame = img
            # 使用 YOLO 检测人脸
            results = self.yolo_model.predict(frame, conf=0.7, verbose=False)
            # 获取检测框信息
            boxes = results[0].boxes
            # 遍历所有检测框
            for box in boxes:
                # 获取检测框坐标
                x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                # 裁剪人脸区域
                face_img = frame[y0:y1, x0:x1]
                # 添加到列表
                face_imgs.append(face_img)

        # 返回包含所有裁剪人脸图像的列表
        return face_imgs

    # 将 OpenCV 图像转换为 PyQt 可显示格式的方法
    def display_original_image(self, img):
        """
        将 OpenCV 图像转换为 PyQt 可显示的格式
        :param img: OpenCV 图像（numpy 数组）
        :return: QPixmap 对象，用于在 PyQt 界面显示
        """
        # BGR 转 RGB（OpenCV 使用 BGR，PyQt 使用 RGB）
        # cv2.cvtColor() 进行颜色空间转换
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 获取图像的高度、宽度和通道数
        # shape 属性返回 (height, width, channels)
        h, w, ch = rgb_img.shape
        # 计算每行占用的字节数
        # 通道数 × 宽度 = 每行的像素数 × 每像素字节数（这里每个通道 1 字节）
        bytes_per_line = ch * w

        # 转换为 QImage
        # QImage() 构造函数创建 Qt 图像对象
        # rgb_img.data 获取图像数据的内存指针
        # w, h 指定宽度和高度
        # bytes_per_line 指定每行的字节跨度
        # QImage.Format_RGB888 指定图像格式为 RGB 888（每个像素 3 字节）
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 从 QImage 创建 QPixmap
        # QPixmap.fromImage() 静态方法，将 QImage 转换为 QPixmap
        # QPixmap 可以直接在 PyQt 的 QLabel 等控件上显示
        pixmap = QPixmap.fromImage(qimg)

        # 返回转换后的 QPixmap 对象
        return pixmap

    # 在图像上绘制人脸检测框的方法
    def draw_face_boxes(self, frame):
        """
        在图像上绘制人脸检测框
        :param frame: OpenCV 图像（numpy 数组）
        :return: 绘制了检测框的图像
        """
        # 复制输入图像，避免修改原始数据
        # copy() 创建图像的深拷贝
        processed_frame = frame.copy()
        # 使用 YOLO 检测人脸
        # predict() 进行目标检测
        # conf=0.7 设置置信度阈值
        # verbose=False 关闭详细输出
        results = self.yolo_model.predict(processed_frame, conf=0.7, verbose=False)
        # 获取检测框信息
        boxes = results[0].boxes
        # 遍历所有检测框
        for box in boxes:
            # 获取检测框的坐标
            # xyxy[0] 获取框的左上角和右下角坐标
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            # 绘制绿色矩形框，线宽 2 像素
            # cv2.rectangle() 绘制矩形
            # (x0, y0) 左上角坐标
            # (x1, y1) 右下角坐标
            # (0, 255, 0) 颜色（BGR 格式，这里是绿色）
            # 2 线条宽度（像素）
            cv2.rectangle(processed_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        # 返回绘制了检测框的图像
        return processed_frame


# Python 脚本的主入口判断
# 只有直接运行此文件时才会执行以下代码
# 如果被其他模块导入，则不会执行
if __name__ == '__main__':
    # 测试代码

    # YOLO 模型权重路径
    # r'' 表示原始字符串，避免转义字符问题
    # 指向本地保存的 YOLO 训练最佳权重文件
    yolo_weights_path = r'C:\Users\Lenovo\Desktop\HQYJ\Facial_Recognition\my_yolo\runs\detect\train\weights\best.pt'

    # 测试图片路径
    # 指向一张用于测试的图片文件
    img = r"C:\Users\Lenovo\Desktop\HQYJ\Facial_Recognition\dataset\images\test\Antony_Leung_0002.jpg"

    # 创建 FaceNet 实例
    # 传入 YOLO 模型权重路径
    face = FaceNet(yolo_weights_path)

    # 检测并裁剪人脸
    # 调用 getfacepos() 方法，传入测试图片路径
    # 返回所有检测到的人脸图像列表
    face_imgs = face.getfacepos(img)

    # 输出检测结果
    # 打印人脸图像列表
    print(face_imgs)
