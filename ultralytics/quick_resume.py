#!/usr/bin/env python3
"""
快速恢复训练脚本
"""

from ultralytics import YOLO

# 加载最后的检查点
model = YOLO('runs/detect/train15/weights/last.pt')

# 恢复训练
print("🚀 恢复训练中...")
results = model.train(resume=True)

print("✅ 训练完成！")
print(f"📁 结果保存在: {results.save_dir}")
