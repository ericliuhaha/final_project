import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv
import math
import pandas as pd

face_box=[]
color = (0, 255, 0)


model = YOLO('runs/detect/train/weights/best.pt')
image_path = '303.jpg'
save_path = 'annotated_test_picture2.jpg'


# 使用 InferenceSlicer 將圖片分割並進行推論
image = cv2.imread(image_path)
slicer = sv.InferenceSlicer(callback=lambda image_slice: sv.Detections.from_ultralytics(model(image_slice)[0]))
detections = slicer(image)
#print(f"Total detections: {len(detections)}")


#偵測臉的function
for i, detection in enumerate(detections):
    xyxy, mask, confidence, class_id, tracker_id, data = detection

    #左上,右下
    x1, y1, x2, y2 = map(int, xyxy)
    if class_id == 1:  # 假設 1 代表臉部
       Area=abs(x2-x1)*abs(y2-y1)
       wait=[x1,y1,x2,y2,Area]
       face_box.append(wait)
       




# 假設 face_box 已經存在，這是包含 x1, y1, x2, y2, area 的列表
xx1, yy1, xx2, yy2, areas = [], [], [], [], []


# 排序並取中位數
face_box = sorted(face_box, key=lambda x: x[4])



def check_ratio(index):
    """檢查 face_box[index] 的寬高比是否符合條件"""
    x1, y1, x2, y2, area = face_box[index]
    height = abs(y2 - y1)
    width = abs(x2 - x1)
    ratio = height / width if width != 0 else 0
    return 1.20 < ratio < 1.70

count = len(face_box)  # 修正 count 計算
mid = count // 2  # 取正確的中間索引
right, left = mid, mid
flag = 0

if not face_box:  # face_box 為空時
    print("face_box 為空，無法繼續執行")
else:
    while right < count or left >= 0:
        if right < count and check_ratio(right):  # 優先檢查右側
            lens = right
            flag = 1
            break
        elif right < count:  # 若沒找到，先移動右指標
            right += 1

        elif right >= (count * 5) // 6 and left >= 0 and check_ratio(left):  # 當右指標超過門檻，開始左側搜尋
            lens = left
            flag = 1
            break
        elif left >= 0:  # 左指標繼續移動
            left -= 1

        elif left < 0 and right < count and check_ratio(right):  # 左指標到底，右指標繼續移動
            lens = right
            flag = 1
            break
        elif left < 0 and right < count:  # 右指標繼續移動
            right += 1
        
        if flag == 1:
            break

# 若沒有符合條件的框，回傳中間索引
if flag == 0:
    lens = mid

print("總框數:", len(face_box))
print("選擇的索引:", lens)
print("選擇的框:", face_box[lens])
print("所有框:", face_box)

    
    

#lens = math.ceil(len(face_box) / 2)  # 中位數的位置

# 篩選
for i in range(len(face_box)):
    x1, y1, x2, y2, area = face_box[i]
    
    # 篩選條件: 面積大小
    if face_box[lens][4] * 0.75 < area and face_box[lens][4] * 1.75 > area:
        # 計算長寬比
        height = abs(y2 - y1)
        width = abs(x2 - x1)
        ratio = height / width if width != 0 else 0


        xx1.append(x1)
        yy1.append(y1)
        xx2.append(x2)
        yy2.append(y2)
        areas.append(area)

        

#抓重疊
overlap=[0]*len(xx1)
for i in range(len(xx1)):
    for j in range(i + 1, len(xx1)):
        
        x1a, y1a, x2a, y2a = xx1[i], yy1[i], xx2[i], yy2[i]
        x1b, y1b, x2b, y2b = xx1[j], yy1[j], xx2[j], yy2[j]

        # 判斷是否重疊
        if not (x2a < x1b or x2b < x1a or y2a < y1b or y2b < y1a):  
            overlap[i]=1
            overlap[j]=1
            
        
for i in range(len(xx1)):
    if overlap[i]==1:   
        cv2.rectangle(image, (xx1[i], yy1[i]), (xx2[i], yy2[i]), (0, 0, 255), 1)  
    else:
        cv2.rectangle(image, (xx1[i], yy1[i]), (xx2[i], yy2[i]), (0, 225, 0), 1)



# 將篩選後的結果轉成 DataFrame
df = pd.DataFrame({
    'x1': xx1,
    'y1': yy1,
    'x2': xx2,
    'y2': yy2,
    'area': areas,
    'overlap':overlap
})

# 儲存為 CSV 檔案
df.to_csv('face_box_data.csv', index=False)

# 顯示圖片
cv2.imshow('Annotated Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()