import torch
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap

from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1


class  FaceNet:
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


        # 判断输入是图片路径还是 numpy 数组（摄像头帧）
        if isinstance(img, str):
            img_path = Path(img)
            # 如果是目录
            if img_path.is_dir():
                for image_path in img_path.glob('*.jpg'):
                    frame = cv2.imread(str(image_path))
                    results = self.yolo_model.predict(frame, conf=0.7, verbose=False)
                    boxes = results[0].boxes
                    for box in boxes:
                        x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                        # face_position = [x0, y0, x1, y1]
                        # all_face_positions.append(face_position)
                        face_img = frame[y0:y1, x0:x1]
                        face_imgs.append(face_img)
            # 如果是单张图片
            elif img_path.is_file():
                frame = cv2.imread(str(img_path))
                results = self.yolo_model.predict(frame, conf=0.7, verbose=False)
                boxes = results[0].boxes
                for box in boxes:
                    x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                    # face_position = [x0, y0, x1, y1]
                    # all_face_positions.append(face_position)
                    face_img = frame[y0:y1, x0:x1]
                    face_imgs.append(face_img)
        # 如果是 numpy 数组（摄像头帧）
        elif isinstance(img, np.ndarray):
            frame = img
            results = self.yolo_model.predict(frame, conf=0.7, verbose=False)
            boxes = results[0].boxes
            for box in boxes:
                x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                # face_position = [x0, y0, x1, y1]
                # all_face_positions.append(face_position)
                face_img = frame[y0:y1, x0:x1]
                face_imgs.append(face_img)

        return face_imgs


    def display_original_image(self, img):
        """显示原始图像"""
        # BGR 转 RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w

        # 转换为 QImage
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        return pixmap






if __name__ == '__main__':
    yolo_weights_path = r'C:\Users\Lenovo\Desktop\HQYJ\Facial_Recognition\my_yolo\runs\detect\train\weights\best.pt'
    img = r"C:\Users\Lenovo\Desktop\HQYJ\Facial_Recognition\dataset\images\test\Antony_Leung_0002.jpg"
    face = FaceNet(yolo_weights_path)
    face_imgs = face.getfacepos(img)
    print(face_imgs)
