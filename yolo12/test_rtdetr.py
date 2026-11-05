import warnings
warnings.filterwarnings("ignore")
from ultralytics import RTDETR

if __name__ == "__main__":
    # 从模型配置文件创建网络
    model = RTDETR("runs/detect/rt-detr-x-DIOR.pt")
    model.val(data='data.yaml',  
              split='test',
              imgsz=640,
              batch=128,
              conf=0.001,
              workers=8,
    )