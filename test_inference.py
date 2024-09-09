import torch
import cv2
import numpy as np

# 加载自定义训练好的YOLOv5模型，替换成你的权重文件路径
model_path = './runs/train/exp14/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# 设置设备为GPU（如果有）或者CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

model.to(device)

# 读取输入图像
image_path = '/root/dataset/glass_data/images/val/Image_20240725165715498.bmp'
img = cv2.imread(image_path)

# 转换为RGB格式（YOLOv5的模型要求输入为RGB格式）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 推理
results = model(img_rgb)

# 打印检测结果
print(results.pandas().xyxy[0])  # 打印结果数据：x1, y1, x2, y2, confidence, class, name

# 可视化检测结果
results.show()  # 使用默认绘图工具显示结果

# 绘制检测框并显示
for result in results.xyxy[0]:
    x1, y1, x2, y2, conf, cls = map(int, result[:6])
    label = f'{model.names[cls]} {conf:.2f}'
    print(f"result: {result}")