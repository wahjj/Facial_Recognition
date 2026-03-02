from ultralytics import YOLO
# 测试图片
model = YOLO(r'C:\Users\Lenovo\Desktop\HQYJ\Facial Recognition\runs\detect\train\weights\best.pt')


results = model.predict(source =r"C:\Users\Lenovo\Desktop\HQYJ\Facial Recognition\dataset\images\test" )

for result in results:
    result.show()

# import cv2
# # 测试视频
#
# cap =  cv2.VideoCapture(0)
#
# while cap.isOpened():
#     ret,frame = cap.read()
#     if ret:
#         results = model.predict(frame)
#         # 将结果绘制在图片上
#         frame = results[0].plot()
#         cv2.imshow('result',frame)
#         cv2.waitKey(1)
#     else:
#         break