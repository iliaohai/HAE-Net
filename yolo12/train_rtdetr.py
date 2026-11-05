warnings.filterwarnings("ignore") 
from ultralytics import RTDETR 
if __name__ == "__main__": 
    # 从模型配置文件创建网络 
    model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-x.yaml") 
    # 加载预训练权重 
    model.load("pre-models/rtdetr-x.pt") 
    model.info() 
    results = model.train(
        data="data.yaml", 
        imgsz=640, 
        epochs=2000, 
        batch=16, 
        workers=2, 
        device=0, 
        optimizer="SGD", 
        amp=True, 
        cache=False, 
    )