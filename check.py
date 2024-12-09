import os

# Đường dẫn tới thư mục chứa dữ liệu khuôn mặt
data_path = "face_dataset"

# Lấy danh sách các thư mục trong 'face_dataset'
folders = os.listdir(data_path)

# In ra danh sách các thư mục
print("Danh sách thư mục trong 'face_dataset':", folders)

# Sắp xếp và in lại thứ tự thư mục
folders.sort()
print("Thứ tự các thư mục OpenCV duyệt qua:", folders)
