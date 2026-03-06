# facenet.py
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from PyQt5.QtGui import QImage, QPixmap


class FaceNet:
    def __init__(self, yolo_model_path):
        self.facenet_model = InceptionResnetV1(pretrained='casia-webface').eval().to('cpu')
        self.yolo_model = YOLO(yolo_model_path)

    def preprocess_face_img(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (160, 160))
        face_tensor = (torch.tensor(face_img).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
        return face_tensor

    def get_embedding(self, face_img):
        face_tensor = self.preprocess_face_img(face_img)
        with torch.no_grad():
            face_embedding = self.facenet_model(face_tensor)
            l2_norm = torch.norm(face_embedding, p=2, dim=1, keepdim=True)
            face_embedding_normalized = face_embedding.div(l2_norm)
        return face_embedding_normalized.cpu().numpy().flatten()

    def detect_faces(self, img):
        """返回人脸框列表和人脸图像列表"""
        results = self.yolo_model.predict(img, conf=0.7, verbose=False)
        boxes = results[0].boxes
        face_imgs = []
        coords = []
        for box in boxes:
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            coords.append((x0, y0, x1, y1))
            face_imgs.append(img[y0:y1, x0:x1])
        return coords, face_imgs

    def draw_boxes(self, img, boxes):
        for (x0, y0, x1, y1) in boxes:
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return img

    @staticmethod
    def pixmap_from_cv(img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)