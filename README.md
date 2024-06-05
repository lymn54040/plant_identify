
# plant_identification

### 安裝套件
  (運行環境 : python = 3.9)
```bash
git clone https://github.com/lymn54040/plant_identification.git
```
### 運行專案

```bash
cd ultralytics
```
```bash
python pred.py
```
## pred.py說明
### 1. 手動設定的部分
- img_path -> 設定欲偵測的圖片位址
- save_dir -> 設定圖片結果儲存位址
### 2. 使用模型預測
- result = pred() -> label和bounding box的結果儲存在result
- 預測結果圖片儲存在save_dir
- 若運行後不想跳出cv2的圖片，可註解掉visualize_labelme_annotations()函式的最後三行
 
...
