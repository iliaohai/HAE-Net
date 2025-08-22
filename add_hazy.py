import os
import numpy as np
import cv2
import random
from tqdm import tqdm  # 进度条显示

def add_hazy(image, beta=None, brightness=None):
    '''
    :param image:       输入图像 (BGR格式)
    :param beta:        雾浓度 (控制能见度)，随机范围 [0.03, 0.08]
    :param brightness:  雾霾亮度 (控制灰度)，随机范围 [0.5, 0.9]
    :return:            加雾后的图像
    '''
    beta = beta if beta is not None else random.uniform(0.06, 0.10)
    brightness = brightness if brightness is not None else random.uniform(0.6, 1.0)
    
    img_f = image.astype(np.float32) / 255.0
    row, col, chs = image.shape
    size = np.sqrt(max(row, col))  
    center = (row // 2, col // 2) 
    y, x = np.ogrid[:row, :col]
    dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    d = -0.04 * dist + size
    td = np.exp(-beta * d)
    img_f = img_f * td[..., np.newaxis] + brightness * (1 - td[..., np.newaxis])
    return np.clip(img_f * 255, 0, 255).astype(np.uint8)

def process_folder(input_folder, output_folder):
    '''
    :param input_folder:  输入图片文件夹路径
    :param output_folder: 输出图片文件夹路径
    '''
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入文件夹中的所有图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        raise FileNotFoundError(f"输入文件夹中没有找到图片文件: {input_folder}")
    
    # 处理每张图片
    for img_name in tqdm(image_files, desc="处理进度"):
        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)
        
        # 读取图片
        image = cv2.imread(input_path)
        if image is None:
            print(f"警告: 无法加载图片 {input_path}, 已跳过")
            continue
        
        # 添加雾霾并保存
        image_fog = add_hazy(image)
        cv2.imwrite(output_path, image_fog)
    
    print(f"\n处理完成! 结果已保存到: {output_folder}")

if __name__ == '__main__':
    # 配置输入输出文件夹路径
    input_dir = "../yolo12/datasets/DIOR/images/test"    # 原始图片文件夹
    output_dir = "../yolo12/datasets/DIOR/images/test-hazy"    # 加雾后图片保存文件夹
    
    # 执行批量处理
    process_folder(input_dir, output_dir)