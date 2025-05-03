import cv2
import time
from ultralytics import YOLO
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from models import transform, EyeCNN, device, model_path

def predict(image_path, eye_model):
    eye_model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = eye_model(image)
        _, predicted = output.max(1)
    
    return "Open" if predicted.item() == 1 else "Closed"

def crop_and_save(left_or_right, a, b, c, d, x, frame):
    cropped_img = frame[b:d, a:c]

    # Use relative path for the output directory
    output_dir = os.path.join(os.path.dirname(__file__), "cut_picture")  # Current directory
    os.makedirs(output_dir, exist_ok=True)

    if left_or_right == 'left':
        output_filename = os.path.join(output_dir, f"output_left_{x}.jpg")
    else:
        output_filename = os.path.join(output_dir, f"output_right_{x}.jpg")

    success = cv2.imwrite(output_filename, cropped_img)
    if not success:
        print(f"Error: 無法儲存圖片 {output_filename}")
    else:
        print(f"圖片已儲存: {output_filename}")

    return output_filename

# 加載 EyeCNN 模型
eye_model = EyeCNN().to(device)
eye_model.load_state_dict(torch.load(model_path, weights_only=True))
eye_model.eval()

# 加載 YOLO 模型
yolo_model = YOLO('runs/detect/train/weights/best.pt')

cap = cv2.VideoCapture(0)

interval = 1
last_time = time.time()

latest_results = None  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if time.time() - last_time >= interval:
        last_time = time.time()
        latest_results = yolo_model(frame)  # 使用 yolo_model 進行偵測

    face_boxes = []

    if latest_results:  
        for result in latest_results:  
            for box in result.boxes:  
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                if conf > 0.3:  
                    if cls == 1:
                        face_boxes.append([x1, y1, x2, y2])

    overlap = [0] * len(face_boxes)
    for i in range(len(face_boxes)):
        for j in range(i + 1, len(face_boxes)):
            x1a, y1a, x2a, y2a = face_boxes[i]
            x1b, y1b, x2b, y2b = face_boxes[j]

            if not (x2a < x1b or x2b < x1a or y2a < y1b or y2b < y1a):  
                overlap[i] = 1
                overlap[j] = 1

    for i in range(len(face_boxes)):
        color = (0, 0, 255) if overlap[i] == 1 else (0, 255, 0)
        x1, y1, x2, y2 = face_boxes[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        face_width = x2 - x1
        face_height = y2 - y1
        eye_y1 = y1 + int(face_height * 0.15)
        eye_y2 = y1 + int(face_height * 0.45)

        left_eye_x1 = x1 + int(face_width * 0.1)
        left_eye_x2 = x1 + int(face_width * 0.45)
        
        right_eye_x1 = x1 + int(face_width * 0.55)
        right_eye_x2 = x1 + int(face_width * 0.9)

        left_eye_path = crop_and_save('left', left_eye_x1, eye_y1, left_eye_x2, eye_y2, i, frame)
        right_eye_path = crop_and_save('right', right_eye_x1, eye_y1, right_eye_x2, eye_y2, i, frame)

        left_result = predict(left_eye_path, eye_model)
        if left_result == 'Open':
            cv2.rectangle(frame, (left_eye_x1, eye_y1), (left_eye_x2, eye_y2), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (left_eye_x1, eye_y1), (left_eye_x2, eye_y2), (0, 0, 255), 2)

        right_result = predict(right_eye_path, eye_model)
        if right_result == 'Open':
            cv2.rectangle(frame, (right_eye_x1, eye_y1), (right_eye_x2, eye_y2), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (right_eye_x1, eye_y1), (right_eye_x2, eye_y2), (0, 0, 255), 2)

    cv2.imshow("YOLO Face & Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
