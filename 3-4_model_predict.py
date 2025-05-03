import pandas as pd
import numpy as np
import cv2
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from models import transform , get_dataloaders , EyeCNN , device, model_path

def predict(image_path):
    model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
    
    return "Open" if predicted.item() == 1 else "Closed"




def crop_and_save(left_or_right, a, b, c, d, x, image_path):
    img = cv2.imread(image_path)
    cropped_img = img[b:d, a:c]

    # 使用相對路徑
    output_dir = os.path.join(os.getcwd(), "cut_picture")  # 取得當前執行目錄並建立 cut_picture 資料夾
    os.makedirs(output_dir, exist_ok=True)  # 確保目錄存在

    if left_or_right == 'left':
        output_filename = os.path.join(output_dir, f"output_left_{x}.jpg")
    else:
        output_filename = os.path.join(output_dir, f"output_right_{x}.jpg")

    success = cv2.imwrite(output_filename, cropped_img)
    if not success:
        print(f"Error: 無法儲存圖片 {output_filename}")
    else:
        print(f"圖片已儲存: {output_filename}")



# 讀取 CSV 檔案並存成 DataFrame
file_path = "eyes_data_with_new_columns.csv"  # 替換為實際的 CSV 檔案路徑
df = pd.read_csv(file_path)


print(df)
ans = df.to_numpy()
print(ans)

# 讀取圖片
image_path = "303.jpg"  # 替換為實際的影像檔案路徑
image = cv2.imread(image_path)  # 讀取圖片

if image is None:
    print(f"Error: 無法讀取圖片 {image_path}")
    exit()


x = 0
right_pred=[]
left_pred=[]

model = EyeCNN().to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()


for a, b, c, d, e, f, g, h, i, j, k, l in ans:
    crop_and_save('left', e, f, g, h, x, image_path)
    crop_and_save('right', i, j, k, l, x, image_path)


    pre = os.path.join('cut_picture', f'output_left_{x}.jpg')
    result = predict(pre)
    print(f"Prediction: {result}")
    if result=='Open':
        left_pred.append(1)
    if result=="Closed":
        crop_and_save('left', e, f+((h-f)//4), g, h+((h-f))//4, x, image_path)
        pre = os.path.join('cut_picture', f'output_left_{x}.jpg')
        result = predict(pre)
        print(f"new_Prediction: {result}")
        if result=='Closed':
            left_pred.append(0)
        else:
            left_pred.append(0.5)



    pre = os.path.join('cut_picture', f'output_right_{x}.jpg')
    result = predict(pre)
    print(f"Prediction: {result}")
    if result=='Open':
        right_pred.append(1)
    if result=="Closed":
        crop_and_save('right', i, j+((l-j)//4), k, l+((l-j)//4), x, image_path)
        pre = os.path.join('cut_picture', f'output_right_{x}.jpg')
        result = predict(pre)
        print(f"new_Prediction: {result}")
        if result=='Closed':
            right_pred.append(0)
        else:
            right_pred.append(0.5)


    x += 1

print(left_pred)
print(right_pred)

xx=0
for a, b, c, d, e, f, g, h, i, j, k, l in ans:
    if left_pred[xx]==1 :
        cv2.rectangle(image, (e,f),(g,h), (0, 225, 0), 1)
    elif left_pred[xx]==0:
        cv2.rectangle(image, (e,f),(g,h), (0, 0, 255), 1)
    else:    #往下predict有偵測到的
        cv2.rectangle(image, (e, f+((h-f)//4)), (g, h+((h-f))//4), (0, 225, 0), 1)
    

    if right_pred[xx]==1 :
        cv2.rectangle(image, (i,j),(k,l), (0, 225, 0), 1)
    elif right_pred[xx]==0:
        cv2.rectangle(image, (i,j),(k,l), (0, 0, 255), 1)
    else:    #往下predict有偵測到的
        cv2.rectangle(image, (i, j+((l-j)//4)), (k, l+((l-j)//4)), (0, 225, 0), 1)
    
    xx+=1

cv2.imshow('Annotated Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()