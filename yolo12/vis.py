import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/train14/weights/best.pt')

# 读取图像

# image_path = 'datasets/HAE-DIOR/images/test/17441.jpg'
# image_path = 'datasets/HAE-DIOR/images/test/17758.jpg'
# image_path = 'datasets/HAE-DIOR/images/test/19396.jpg'
# image_path = 'datasets/HAE-DIOR/images/test/19809.jpg'
image_path = '001757.jpg'



# 推理
results = model(image_path)[0]  # 只取第一个图像的结果

# 读取原图
image = cv2.imread(image_path)

# 遍历检测框
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = f"{model.names[cls_id]} {conf:.2f}"

    # 绘制边框和标签（小字体）
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# 保存结果图
cv2.imwrite("vis_result.jpg", image)