#!/usr/bin/env python3
"""
恢复中断的YOLO训练脚本
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def resume_training():
    """恢复中断的训练"""
    
    print("🔄 恢复YOLO训练")
    print("=" * 50)
    
    # 检查点文件路径
    last_checkpoint = "/home/hllqk/projects/yolo-mecd/ultralytics/runs/detect/train15/weights/last.pt"
    
    # 检查检查点文件是否存在
    if not os.path.exists(last_checkpoint):
        print(f"❌ 检查点文件不存在: {last_checkpoint}")
        return False
    
    print(f"✅ 找到检查点文件: {last_checkpoint}")
    
    try:
        # 加载模型从检查点
        print("📦 加载模型检查点...")
        model = YOLO(last_checkpoint)
        
        print("📊 检查点信息:")
        print(f"  - 模型类型: {model.task}")
        print(f"  - 类别数量: {len(model.names)}")
        print(f"  - 类别名称: {list(model.names.values())}")
        
        # 恢复训练
        print("\n🚀 恢复训练...")
        print("训练参数:")
        print("  - 目标epochs: 200")
        print("  - 当前进度: ~175/200 epochs")
        print("  - 剩余epochs: ~25")
        print("  - 数据集: citdet.yaml")
        print("  - 图像尺寸: 1280")
        print("  - 批次大小: 4")
        
        # 使用resume=True恢复训练
        results = model.train(
            resume=True,  # 关键参数：恢复训练
            verbose=True
        )
        
        print("\n✅ 训练恢复成功！")
        return True
        
    except Exception as e:
        print(f"❌ 恢复训练失败: {str(e)}")
        return False

def check_training_status():
    """检查训练状态"""
    print("\n📊 训练状态检查")
    print("-" * 30)
    
    # 检查结果文件
    results_file = "/home/hllqk/projects/yolo-mecd/ultralytics/runs/detect/train15/results.csv"
    if os.path.exists(results_file):
        # 读取最后几行
        with open(results_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) > 1:
            # 获取最后一行数据
            last_line = lines[-2].strip()  # -2因为最后一行可能是空行
            if last_line:
                parts = last_line.split(',')
                if len(parts) >= 9:
                    epoch = parts[0]
                    mAP50 = parts[7]
                    mAP50_95 = parts[8]
                    
                    print(f"✅ 最后训练epoch: {epoch}")
                    print(f"✅ 当前mAP50: {mAP50}")
                    print(f"✅ 当前mAP50-95: {mAP50_95}")
                    
                    # 计算剩余epochs
                    remaining = 200 - int(epoch)
                    print(f"📈 剩余epochs: {remaining}")
                    
                    if remaining > 0:
                        print(f"💡 建议继续训练 {remaining} 个epochs")
                    else:
                        print("🎉 训练已完成！")
    
    # 检查权重文件
    weights_dir = "/home/hllqk/projects/yolo-mecd/ultralytics/runs/detect/train15/weights"
    if os.path.exists(weights_dir):
        weight_files = os.listdir(weights_dir)
        print(f"\n📁 可用权重文件: {len(weight_files)}")
        for file in sorted(weight_files):
            file_path = os.path.join(weights_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file}: {size_mb:.1f} MB")

def main():
    """主函数"""
    print("🍊 YOLO训练恢复工具")
    print("=" * 50)
    
    # 检查当前状态
    check_training_status()
    
    # 询问是否继续训练
    print("\n" + "=" * 50)
    response = input("是否要恢复训练？(y/n): ").lower().strip()
    
    if response in ['y', 'yes', '是', '1']:
        print("\n🚀 开始恢复训练...")
        success = resume_training()
        
        if success:
            print("\n🎉 训练恢复完成！")
            print("📁 结果保存在: runs/detect/train15/")
            print("🏆 最佳模型: runs/detect/train15/weights/best.pt")
        else:
            print("\n❌ 训练恢复失败，请检查错误信息")
    else:
        print("\n👋 取消恢复训练")
        print("\n💡 手动恢复命令:")
        print("   from ultralytics import YOLO")
        print("   model = YOLO('runs/detect/train15/weights/last.pt')")
        print("   model.train(resume=True)")

if __name__ == "__main__":
    main()
