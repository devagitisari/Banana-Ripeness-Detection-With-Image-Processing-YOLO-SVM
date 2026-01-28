import cv2

print("=== Mencari kamera aktif di index 0–5 ===")
for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Kamera aktif di index {i}")
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(1000)  # tampilkan 1 detik per kamera
            cv2.destroyAllWindows()
        cap.release()
    else:
        print(f"❌ Kamera di index {i} tidak tersedia")

cv2.destroyAllWindows()
