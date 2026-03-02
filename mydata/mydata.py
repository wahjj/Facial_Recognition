import os
import shutil


files = os.listdir('./lfw')
file_abs = os.path.abspath('./lfw')

for file in files:
    file_path = os.path.join(file_abs,file)
    if os.path.isdir(file_path):
        imgFiles = os.listdir(file_path)
        for img in imgFiles:
            shutil.copy(os.path.join(file_path,img),'../data')
            print(f"{os.path.join(file_path,img)}复制成功")