import torch
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1


def preprocess_face_img(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    size = (160, 160)
    face_img = cv2.resize(face_img, size)
    face_tensor = (torch.tensor(face_img).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
    return face_tensor


def getFacePos(yolov_weights_path, img):
    yolov11_model = YOLO(yolov_weights_path)
    img_path = Path(img)
    face_imgs = []
    for image_path in img_path.glob('*.jpg'):
        frame = cv2.imread(str(image_path))
        results = yolov11_model.predict(frame, conf=0.7)  # 添加置信度阈值
        boxes = results[0].boxes
        if boxes is not None:  # 检查是否有检测到框
            for box in boxes:
                x0, y0, x1, y1 = box.xyxy[0].tolist()
                cls = box.cls[0].tolist()  # 修正拼写错误
                if cls == 0:  # 假设0是人脸类别
                    face_img = frame[int(y0):int(y1), int(x0): int(x1)]
                    face_imgs.append(face_img)  # 收集所有人脸图像
    return face_imgs  # 返回所有人脸图像列表


def facenet(face_img):  # 修改参数名为 face_img
    face_tensor = preprocess_face_img(face_img=face_img)
    facenet_model = InceptionResnetV1(pretrained='casia-webface').eval().to('cpu')
    with torch.no_grad():
        face_embedding = facenet_model(face_tensor)
        l2_norm = torch.norm(face_embedding, p=2, dim=1, keepdim=True)
        face_embedding_normalized = face_embedding.div(l2_norm)
    return face_embedding_normalized.cpu().numpy()


if __name__ == '__main__':
    yolov_weights_path = r'C:\Users\Lenovo\Desktop\HQYJ\Facial Recognition\my_yolo\runs\detect\train\weights\best.pt'
    img = r"C:\Users\Lenovo\Desktop\HQYJ\Facial Recognition\dataset\images\test"
    face_imgs = getFacePos(yolov_weights_path, img)
    if face_imgs:  # 检查是否检测到人脸
        for i, pos in enumerate(face_imgs):
            print(f"人脸位置区域 {i}: shape {pos}")
            vec = facenet(pos)
            print(f"向量长度 {i}: {vec}")
    else:
        print("未检测到任何人脸")