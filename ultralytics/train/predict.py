import numpy as np
import cv2
from mss import mss
from PIL import Image
from Xlib import display, X
from ultralytics import YOLO

# 加载YOLO模型
model = YOLO("/home/hllqk/projects/yolo-mecd/ultralytics/train/best.pt")

# 获取屏幕尺寸
def get_screen_size():
    d = display.Display().screen().root
    return d.get_geometry().width, d.get_geometry().height

screen_width, screen_height = get_screen_size()

# 设置捕获整个屏幕
monitor = {
    "top": 0,
    "left": 0,
    "width": screen_width,
    "height": screen_height,
    "mon": 0
}

# 创建屏幕捕获对象
sct = mss()

# 创建透明覆盖窗口
cv2.namedWindow("YOLO Overlay", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLO Overlay", cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty("YOLO Overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.resizeWindow("YOLO Overlay", screen_width, screen_height)

while True:
    # 捕获屏幕内容
    sct_img = sct.grab(monitor)
    img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
    img_np = np.array(img)
    
    # 使用YOLO模型进行预测
    results = model.predict(img_np, imgsz=1024, conf=0.2)
    
    # 创建透明覆盖层
    overlay = np.zeros_like(img_np)
    
    # 绘制检测结果
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标
        classes = result.boxes.cls.cpu().numpy()  # 获取类别
        confidences = result.boxes.conf.cpu().numpy()  # 获取置信度
        
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            
            # 绘制边界框
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制标签背景
            cv2.rectangle(overlay, (x1, y1 - 20), (x1 + len(label) * 10, y1), (0, 255, 0), -1)
            # 绘制标签文本
            cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 显示透明覆盖层
    cv2.imshow("YOLO Overlay", overlay)
    
    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()