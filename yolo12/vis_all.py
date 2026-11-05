import os
import cv2
from ultralytics import YOLO

# 用户配置
input_folder = 'vis_input/'  # 输入图片文件夹
output_folder = 'vis_output/'               # 输出结果文件夹
suffix = 'YOLOv8n'                            # 输出文件后缀（如 "17441-detected.jpg"）
model_path = 'runs/detect/train35/weights/best.pt'  # YOLO模型路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 加载模型
model = YOLO(model_path)

# 遍历输入文件夹的所有图片
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # 支持常见图片格式
        image_path = os.path.join(input_folder, filename)
        
        # 推理
        results = model(image_path)[0]
        image = cv2.imread(image_path)

        # 绘制检测框和标签
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 生成输出文件名（如 "17441-detected.jpg"）
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}-{suffix}{ext}"
        output_path = os.path.join(output_folder, output_filename)

        # 保存结果
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")

print("处理完成！")