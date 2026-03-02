from ultralytics import YOLO
# 测试图片
model = YOLO("yolo11n.pt")

if __name__ == '__main__':
    results = model.train(data="../dataset/mydata.yaml",epochs=100)