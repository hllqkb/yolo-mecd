from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo11s.pt")  # 使用yolov8s.pt可能更常见

# 训练配置
results = model.train(
    data="/home/hllqk/projects/yolo-mecd/ultralytics/train/data.yaml",
    epochs=100,
    imgsz=1024,
    save=True,  # 保存训练结果
    save_period=-1,  # 不保存中间周期模型
    save_json=True,  # 保存结果到JSON
    save_hybrid=True,  # 保存混合版本的标签
    single_cls=False,  # 多类训练
    exist_ok=True,  # 允许覆盖现有项目
    name='yolo11s_custom',  # 项目名称
    patience=30,  # 早停耐心值
    batch=4,  # 批量大小
    device='0',  # 使用GPU 0
    workers=4,  # 数据加载线程数
    optimizer='auto',  # 自动选择优化器
    lr0=0.01,  # 初始学习率
    lrf=0.01,  # 最终学习率
    warmup_epochs=3,  # 学习率预热
    box=7.5,  # 框损失权重
    cls=0.5,  # 分类损失权重
    dfl=1.5,  # dfl损失权重
    label_smoothing=0.0,  # 标签平滑
    deterministic=False,
    plots=True  # 保存训练曲线图
)