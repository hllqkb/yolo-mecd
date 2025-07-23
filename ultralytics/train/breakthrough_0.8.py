import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
from ultralytics import YOLO

def setup():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 配置
breakthrough_config = {
    # 数据配置
    'data': '/home/hllqk/projects/yolo-mecd/ultralytics/train/citdet.yaml',  # 数据集配置文件路径
    
    # 训练参数
    'epochs': 200,          # 训练总轮数(先减少轮数进行测试)
    'batch': 4,             # 批次大小(减小4以避免显存不足问题)
    'imgsz': 1280,          # 输入图像分辨率(使 P3、P4 和 P5 检测头对应的特征图尺寸分别达到 160×160、80×80 和 40×40，利于小目标检测)
    
    # 优化器设置
    'optimizer': 'SGD',     # 使用SGD优化器(更稳定但可能收敛慢)
    'lr0': 0.01,            # 初始学习率(标准设置,小目标检测建议稍高)
    'lrf': 0.01,            # 最终学习率(与初始相同，表示不使用衰减)
    'momentum': 0.937,      # SGD动量参数(帮助加速收敛)
    'weight_decay': 0.0005, # 权重衰减(L2正则化系数)
    
    # 学习率预热设置
    'warmup_epochs': 3,     # 学习率预热轮数(减少热身期)
    'warmup_momentum': 0.8, # 预热阶段的动量值
    'warmup_bias_lr': 0.1,  # 预热阶段的偏置学习率
    
    # 损失函数权重
    'box': 7.5,             # 边界框回归损失权重(适中值)
    'cls': 0.5,             # 分类损失权重(平衡设置)
    'dfl': 1.5,             # 分布焦点损失权重(标准设置)
    
    # 数据增强 - 基础设置
    'augment': True,        # 启用数据增强
    'mosaic': 0.5,          # 马赛克增强概率(标准50%)
    'mixup': 0.0,           # 禁用mixup增强(避免复杂问题)
    'copy_paste': 0.0,      # 禁用复制粘贴增强(避免潜在问题)
    
    # 几何变换增强
    'degrees': 0.0,         # 图像旋转角度范围(禁用旋转保护小目标)
    'translate': 0.02,      # 图像平移比例(极小值保持位置稳定)
    'scale': 0.9,           # 图像缩放范围(0.9表示90%-100%缩放)
    'shear': 0.0,           # 图像剪切变换(禁用)
    'perspective': 0.0,     # 透视变换强度(禁用)
    'flipud': 0.0,          # 上下翻转概率(禁用)
    'fliplr': 0.5,          # 左右翻转概率(保持50%概率)
    
    # 颜色空间增强
    'hsv_h': 0.005,         # 色调变化幅度(极小变化)
    'hsv_s': 0.4,           # 饱和度变化幅度(适中)
    'hsv_v': 0.2,           # 亮度变化幅度(适中)
    
    # 训练策略
    'device': 0,            # 使用GPU 0
    'workers': 2,           # 数据加载线程数(减少以避免加载问题)
    'single_cls': False,    # 不使用单类别模式
    'iou_type': 'CIoU',     # 使用CIoU损失(最精确的IoU计算方式)
    'close_mosaic': 20,     # 最后10轮关闭马赛克增强
    'patience': 50,         # 早停耐心轮数(50轮val无改善则停止)
    'amp': True,            # 启用自动混合精度训练
    'cache': False,         # 禁用数据集缓存(避免内存问题)
    'multi_scale': False,   # 禁用多尺度训练(避免复杂性)
    'rect': False,          # 禁用矩形训练
    'cos_lr': False,        # 禁用余弦学习率调度(使用线性更稳定)
    'label_smoothing': 0.0, # 禁用标签平滑
    'nbs': 64,              # 名义批次大小(用于梯度累积)
    
    # 掩码相关参数(实例分割时使用)
    'overlap_mask': True,   # 重叠掩码处理
    'mask_ratio': 4,        # 掩码下采样比率
    
    # 正则化
    'dropout': 0.0,         # 禁用dropout
    
    # 验证与可视化
    'val': True,            # 启用验证
    'plots': True,          # 启用训练过程绘图
    'save_period': 25,      # 每25轮保存一次模型
    'verbose': True,        # 启用详细输出
    
    # 检测参数(针对小目标优化)
    'conf': 0.0005,         # 置信度阈值(极低以检测更多目标)
    'iou': 0.5,             # NMS的IoU阈值(较低以保留更多检测)
    'max_det': 1000,        # 每张图像最大检测数(大幅增加)
    
    # 高级增强技巧
    'erasing': 0.0,         # 禁用随机擦除增强
    'crop_fraction': 1.0,   # 图像裁剪比例(1.0表示不裁剪)
}
def train_breakthrough(resume_path=None):
    # 使用高级架构
    model = YOLO('/home/hllqk/projects/yolo-mecd/ultralytics/ultralytics/cfg/models/11/YOLO11_FINALLY.yaml').load('/home/hllqk/projects/yolo-mecd/ultralytics/yolo11s.pt')
    # model=  YOLO('/home/hllqk/projects/yolo-mecd/ultralytics/yolo11m.pt')
    if resume_path:
        breakthrough_config['resume'] = resume_path 
    # 开始训练
    print("开始训练...")
    results = model.train(**breakthrough_config)
    print("=== 训练完成 ===")
    return model, results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint (e.g., runs/detect/train15/weights/last.pt)')
    args = parser.parse_args()
    model, results = train_breakthrough(resume_path=args.resume)