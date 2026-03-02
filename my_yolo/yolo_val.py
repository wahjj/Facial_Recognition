from ultralytics import YOLO
# 测试图片
model = YOLO(r'C:\Users\Lenovo\Desktop\HQYJ\Facial Recognition\runs\detect\train\weights\best.pt')

if __name__ == '__main__':
    results = model.val(data="../dataset/mydata.yaml")