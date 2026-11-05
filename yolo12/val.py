import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('runs/detect/train56/weights/best.pt')  
    model.val(data='data.yaml',  
              split='test',
              imgsz=640,
              batch=12,
              conf=0.001,
              workers=8,
    )