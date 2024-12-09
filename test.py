import cv2

# Tải mô hình đã huấn luyện
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer.yml")

# Bản đồ nhãn
label_map = {0: "Chi Von", 1: "HIEUTHUHAI"}  # Thay nhãn phù hợp với người dùng

# Khởi tạo webcam và bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_resized = cv2.resize(roi_gray, (200, 200))

        # Nhận diện khuôn mặt
        label_id, confidence = recognizer.predict(roi_gray_resized)
        label_text = label_map.get(label_id, "Unknown")

        # Hiển thị kết quả
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label_text} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'):  # Nhấn 'q' để thoát
        break

cap.release()
cv2.destroyAllWindows()
