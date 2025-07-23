import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
 
if __name__ == '__main__':
    model = YOLO('./ultralytics/cfg/models/11/YOLO11_EMA.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=4, 
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )