#import ultralytics
from ultralytics import YOLO

#
model_yaml = "C:/Users/User/Desktop/plants/ultralytics/ultralytics/cfg/models/v8/yolov8n.yaml"
data_yaml = "D:\plants\plant_label4\data.yaml"
pred_img = "D:\plants\蘭花楹\IMG20240508135645.jpg"
pt_dir = "C:/Users/User/Desktop/stawberry/ultralytics/runs/detect/train11/weights/best.pt"
#-----------------------------------------------------
def train_model():
    # 加载模型
    #model = YOLO("yolov8n.yaml")
    #model = YOLO(model_yaml)  # 从头开始构建新模型
    #model = YOLO(pt_dir)  # 加载预训练模型（建议用于训练）
    model = YOLO(model_yaml).load(pt_dir) #####try
    # 使用模型
    model.train(data=data_yaml, epochs=300, imgsz=640)  # 训练模型
    metrics = model.val()  # 在验证集上评估模型性能
    #results = model(pred_img)  # 对图像进行预测
    success = model.export()  # 将模型导出为 ONNX 格式 format="onnx"
    return model, success

if __name__ == '__main__':
    # 使用模型
    model, export = train_model()
    
    #model.train(data=yaml, epochs=3)  # 训练模型
    #metrics = model.val()  # 在验证集上评估模型性能
    #results = model(pred_img)  # 对图像进行预测
    #success = model.export()  # 将模型导出为 ONNX 格式 format="onnx"