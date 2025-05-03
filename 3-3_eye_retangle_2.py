import pandas as pd
import cv2

# 讀取 CSV 檔案並存成 DataFrame
file_path = "eyes_data.csv"  # 替換為實際的 CSV 檔案路徑
df = pd.read_csv(file_path)

# 打印 DataFrame 和 numpy 陣列檢查
print(df)
ans = df.to_numpy()
print(ans)

# 讀取圖片
image_path = "303.jpg"  # 替換為實際的影像檔案路徑
image = cv2.imread(image_path)  # 讀取圖片

# 檢查圖片是否成功讀取
if image is None:
    print(f"Error: 無法讀取圖片 {image_path}")
    exit()

aa,bb,cc,dd,ee,ff,gg,hh=[],[],[],[],[],[],[],[]
# 遍歷每一個矩形框
for a, b, c, d in ans:
    
    diff=(((a+(c-a)//2)-a)-(d-b))//2
    # 左邊框（正方形）
    left_xx1 = a+diff
    left_yy1 = b
    left_xx2 = (a+(c-a)//2)-diff
    left_yy2 = d 

    # 右邊框（正方形）
    right_xx1 = (a+(c-a)//2)+diff
    right_yy1 = b
    right_xx2 = c-diff
    right_yy2 = d 

    aa.append(left_xx1)
    bb.append(left_yy1)
    cc.append(left_xx2)
    dd.append(left_yy2)
    ee.append(right_xx1)
    ff.append(right_yy1)
    gg.append(right_xx2)
    hh.append(right_yy2)


    # 繪製矩形框，檢查結果
    cv2.rectangle(image, (left_xx1, left_yy1), (left_xx2, left_yy2), (0, 255, 0), 2)  # 綠色框
    cv2.rectangle(image, (right_xx1, right_yy1), (right_xx2, right_yy2), (0, 0, 255), 2)  # 紅色框

# 顯示圖片
cv2.imshow('Annotated Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


df['left_xx1'] = aa
df['left_yy1'] = bb
df['left_xx2'] = cc
df['left_yy2'] = dd
df['right_xx1'] = ee
df['right_yy1'] = ff
df['right_xx2'] = gg
df['right_yy2'] = hh

# 儲存回原始 CSV 檔案
output_file = "eyes_data_with_new_columns.csv"
df.to_csv(output_file, index=False)

