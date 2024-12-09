import cv2
import numpy as np
import os

# Đường dẫn đến thư mục dữ liệu
data_path = "face_dataset"

# Khởi tạo mô hình LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}

label_id = 0
for person_name in os.listdir(data_path):
    label_map[label_id] = person_name
    person_path = os.path.join(data_path, person_name)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(label_id)
    label_id += 1

faces = np.array(faces)
labels = np.array(labels)

# Huấn luyện mô hình
recognizer.train(faces, labels)

# Lưu mô hình vào file
recognizer.save("face_recognizer.yml")
print("Huấn luyện mô hình LBPH thành công!")
