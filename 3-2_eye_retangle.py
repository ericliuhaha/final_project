import pandas as pd
import cv2

# 讀取 CSV 檔案並存成 DataFrame
file_path = "face_box_data.csv"  # 替換為實際的 CSV 檔案路徑
df = pd.read_csv(file_path)
image_path = "303.jpg"  # 替換為實際的影像檔案路徑

# 將影像讀入為 NumPy 陣列
image = cv2.imread(image_path)

x1=[]
y1=[]
x2=[]
y2=[]



if image is None:
    print("影像讀取失敗，請確認檔案路徑。")
else:
    # 查看 DataFrame 的內容
    print(df)

    ans = df.to_numpy()
    print(ans)

    for a, b, c, d, e ,f in ans:
        if f==1:
            continue
        x_gap = abs(c - a) / 10
        y_gap = abs(d - a) / 10
        right = (int(a), int(b+(d-b)*0.05))
        left = (int(c), int(d-(d-b)*0.6))
        
        x1.append(right[0])
        y1.append(right[1])
        x2.append(left[0])
        y2.append(left[1])
        # 繪製矩形框
        cv2.rectangle(image, left, right, (0, 255, 0), 1)

    # 顯示結果
    cv2.imshow('Annotated Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

df = pd.DataFrame({
    'new_x1': x1,
    'new_y1': y1,
    'new_x2': x2,
    'new_y2': y2
})

# 將 DataFrame 存成 CSV 檔案
df.to_csv('eyes_data.csv',index=False)    