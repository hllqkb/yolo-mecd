import sys
from pathlib import Path
from monai.utils import set_determinism
set_determinism(seed=3407) # 3407 is all you need
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
    'epochs': 50,          # 训练总轮数(建议至少两百轮，目前训练是到两百轮后收敛)
    'batch': 4,             # 批次大小(减小4以避免显存不足问题)
    'imgsz': 1280,          # 输入图像分辨率(使 P3、P4 和 P5 检测头对应的特征图尺寸分别达到 160×160、80×80 和 40×40，利于小目标检测)
    
    # 优化器设置
    'optimizer': 'SGD',     # 使用SGD优化器(更稳定但可能收敛慢)
    'lr0': 0.01,            # 初始学习率(标准设置,小目标检测建议稍高)
    'lrf': 0.01,            # 最终学习率(与初始相同，表示不使用衰减)
    'momentum': 0.937,      # SGD动量参数(帮助加速收敛)
    'weight_decay': 0.0005, # 权重衰减(L2正则化系数)
    
    # 损失函数权重
    'box': 7.5,             # 边界框回归损失权重(适中值)
    'cls': 0.5,             # 分类损失权重(平衡设置)
    'dfl': 1.5,             # 分布焦点损失权重(标准设置)
    
    # 数据增强 - 基础设置
    'augment': True,        # 启用数据增强
    
    # 训练策略
    'device': 0,            # 使用GPU 0
    'workers': 2,           # 数据加载线程数(减少以避免加载问题)
    'single_cls': False,    # 不使用单类别模式
    'iou_type': 'CIoU',     # 使用CIoU损失(最精确的IoU计算方式)
    'close_mosaic': 20,     # 最后10轮关闭马赛克增强
    'patience': 50,         # 早停耐心轮数(50轮val无改善则停止)
    'amp': True,            # 启用自动混合精度训练
    'cache': False,         # 禁用数据集缓存(避免内存问题)
    # 正则化
    'dropout': 0.0,         # 禁用dropout
    # 验证与可视化
    'val': True,            # 启用验证
    'plots': True,          # 启用训练过程绘图
    'save_period': 25,      # 每25轮保存一次模型
    'verbose': True,        # 启用详细输出
    # 检测参数(针对小目标优化)
    'conf': 0.0005,         # 置信度阈值(极低以检测更多目标)
}
def train_breakthrough(resume_path=None):
    # 使用高级架构
    model = YOLO('/home/hllqk/projects/yolo-mecd/ultralytics/ultralytics/cfg/models/11/YOLO11_FINALLY.yaml').load('/home/hllqk/projects/yolo-mecd/ultralytics/yolo11s.pt')
    # model=  YOLO('/home/hllqk/projects/yolo-mecd/ultralytics/yolo11m.pt')
    # 使用Yolo12l模型为基准
    # model = YOLO('/home/hllqk/projects/yolo-mecd/ultralytics/ultralytics/cfg/models/11/YOLO11_FINALLY.yaml').load('/home/hllqk/projects/yolo-mecd/yolo12l.pt')
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