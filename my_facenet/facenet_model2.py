import torch
from pathlib import Path
import cv2

from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

facenet_model = InceptionResnetV1(pretrained='casia-webface').eval().to('cpu')

def preprocess_face_img(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (160, 160))
    face_tensor = (torch.tensor(face_img).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
    return face_tensor


def getfacepos(yolo_weights_path, img):
    yolo11_model = YOLO(yolo_weights_path)
    img_path = Path(img)
    face_imgs = []
    all_face_positions = []

    for image_path in img_path.glob('*.jpg'):
        frame = cv2.imread(str(image_path))
        results = yolo11_model.predict(frame, conf=0.7, verbose=False)
        boxes = results[0].boxes
        for box in boxes:
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            face_position = [x0, y0, x1, y1]
            all_face_positions.append(face_position)
            face_img = frame[y0:y1, x0: x1]
            cv2.imshow('Face',face_img)
            cv2.waitKey(1000)
            face_imgs.append(face_img)
    return face_imgs, all_face_positions


def facenet(face_img):
    face_tensor = preprocess_face_img(face_img=face_img)
    with torch.no_grad():
        face_embedding = facenet_model(face_tensor)
        l2_norm = torch.norm(face_embedding, p=2, dim=1, keepdim=True)
        face_embedding_normalized = face_embedding.div(l2_norm)
    return face_embedding_normalized.cpu().numpy()


if __name__ == '__main__':
    yolo_weights_path = r'C:\Users\Lenovo\Desktop\HQYJ\Facial Recognition\my_yolo\runs\detect\train\weights\best.pt'
    img = r"C:\Users\Lenovo\Desktop\HQYJ\Facial Recognition\dataset\images\test"
    face_imgs,all_face_positions = getfacepos(yolo_weights_path, img)
    for i, pos in enumerate(face_imgs):
        print(f"人脸位置区域 {all_face_positions[i]}")
        vec = facenet(pos)
        print(f"向量长度 {i}: {vec}")