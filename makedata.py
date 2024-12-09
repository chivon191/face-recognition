import cv2
import os

# Tạo thư mục lưu trữ ảnh
person_name = "Chi Von"  # Đổi tên theo người dùng
data_path = "face_dataset"  # Thư mục chính chứa dữ liệu
save_path = os.path.join(data_path, person_name)
os.makedirs(save_path, exist_ok=True)

# Khởi tạo webcam và bộ phát hiện khuôn mặt
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (200, 200))  # Resize ảnh về kích thước 200x200
        cv2.imwrite(f"{save_path}/{count}.jpg", face_resized)  # Lưu ảnh
        count += 1

    cv2.imshow("Collecting Faces", frame)
    if cv2.waitKey(1) == ord('q') or count >= 100:  # Lưu 100 ảnh hoặc nhấn 'q' để thoát
        break

cap.release()
cv2.destroyAllWindows()
print(f"Đã lưu {count} ảnh khuôn mặt vào thư mục: {save_path}")