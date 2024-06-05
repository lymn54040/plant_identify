from ultralytics import YOLO
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy

#setting image path
img_path = "./img4.jpg"
save_dir = "./"

#
label_dict = {0: '黃金側柏', 1: '阿勃勒', 2: '琉球松', 3: '黃椰子', 4: '楓香', 5: '榕樹', 6: '熊掌櫚', 7: '蒲葵', 8: '緬梔', 9: '蔥蘭', 10: '龍舌蘭', 11: '刺桐', 12: '龍柏', 13: '鵝掌藤', 14: '黑松', 15: '虎刺梅', 
              16: '五彩千年木', 17: '繁星花', 18: '牽牛花', 19: '風車蓮', 20: '虎尾蘭', 21: '翠蘆莉', 22: '蘭花楹', 23: '變葉木', 24: '九蔥', 25: '台灣欒樹', 26: '紅葉鐵莧', 27: '福木', 28: '黃金葛', 29: '日日春', 30: '白鶴芋', 
              31: '吊竹草', 32: '朱蕉', 33: '南洋杉', 34: '黑松', 35: '杜鵑', 36: '木棉', 37:'鳳凰木', 38: '黃花蜜菜', 39: '馬蘭', 40: '白千層', 41: '七里香'}

def visualize_labelme_annotations(img_path, result, label):
    img_ = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("font/simsun.ttc", 90, encoding="utf-8")

    boxes = result[0].boxes
    for i in range(len(boxes.cls)):
        label = label_dict[int(boxes.cls[i])] #pred label
        
        x0, y0, x1, y1 = boxes.xyxy[i] 
        draw.rectangle([(int(x0), int(y0)), (int(x0) + len(label) * 100, int(y0) + 100)],fill='blue', outline='blue', width=10)
        draw.rectangle([(int(x0), int(y0)), (int(x1), int(y1))], outline='blue', width=10)
        draw.text((int(x0), int(y0)), label, (255, 255, 255), font=fontText)

    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
    img_name = os.path.basename(img_path.replace('.jpg', '_pred.jpg'))
    cv2.imwrite(save_dir + img_name, img) 
    print("predict result save at: ", save_dir + img_name)
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pred():
    # 加载模型
    pt_dir = "./best.pt" 
    model = YOLO(pt_dir)  
    results = model(img_path)  # 對圖像進行預測

    return results

if __name__ == '__main__':
    result = pred()
    boxes = result[0].boxes 
    for i in range(len(boxes.cls)):
        cls = label_dict[int(boxes.cls[i])] #pred label
        #print(boxes.xyxy, boxes.xywh, boxes.conf)
        print('Detection result is:', cls)

    visualize_labelme_annotations(img_path, result, cls)
