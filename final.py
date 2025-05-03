import cv2
import time
from ultralytics import YOLO

# 加載 YOLOv8 模型
model = YOLO('runs/detect/train/weights/best.pt')

# 開啟攝影機
cap = cv2.VideoCapture(0)

# 設定偵測間隔 (秒)
interval = 1  
last_time = time.time()

# 儲存最新的偵測結果
latest_results = None  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 檢查是否達到時間間隔
    if time.time() - last_time >= interval:
        last_time = time.time()  # 更新時間
        latest_results = model(frame)  # 執行 YOLO 偵測，更新結果

    face_boxes = []  # 存儲人臉框

    if latest_results:  
        for result in latest_results:  
            for box in result.boxes:  
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                if conf > 0.3:  # 只考慮信心分數 > 0.3 的偵測結果
                    if cls == 1:  # 人臉
                        face_boxes.append([x1, y1, x2, y2])

    # 偵測重疊框（僅針對人臉）
    overlap = [0] * len(face_boxes)
    for i in range(len(face_boxes)):
        for j in range(i + 1, len(face_boxes)):
            x1a, y1a, x2a, y2a = face_boxes[i]
            x1b, y1b, x2b, y2b = face_boxes[j]

            # 判斷是否重疊
            if not (x2a < x1b or x2b < x1a or y2a < y1b or y2b < y1a):  
                overlap[i] = 1
                overlap[j] = 1

    # 繪製人臉框（紅色表示重疊，綠色表示不重疊）
    for i in range(len(face_boxes)):
        color = (0, 0, 255) if overlap[i] == 1 else (0, 255, 0)  # 紅色（重疊）或綠色（不重疊）
        x1, y1, x2, y2 = face_boxes[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 計算眼睛框
        face_width = x2 - x1
        face_height = y2 - y1
        eye_y1 = y1 + int(face_height * 0.15)  # 眼睛約在臉部 30% 高度處
        eye_y2 = y1 + int(face_height * 0.45)  # 眼睛框高度

        # 左眼框
        left_eye_x1 = x1 + int(face_width * 0.1)
        left_eye_x2 = x1 + int(face_width * 0.45)
        cv2.rectangle(frame, (left_eye_x1, eye_y1), (left_eye_x2, eye_y2), (255, 255, 0), 2)  # 藍色

        # 右眼框
        right_eye_x1 = x1 + int(face_width * 0.55)
        right_eye_x2 = x1 + int(face_width * 0.9)
        cv2.rectangle(frame, (right_eye_x1, eye_y1), (right_eye_x2, eye_y2), (255, 255, 0), 2)  # 藍色

    # 顯示畫面
    cv2.imshow("YOLO Face & Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # 按 'q' 退出
        break

cap.release()
cv2.destroyAllWindows()