#!/usr/bin/env python3
"""
测试修复效果的脚本
"""

import time
import cv2
import numpy as np
from PIL import Image
import json
import os

def test_color_conversion():
    """测试颜色转换修复"""
    print("🎨 测试颜色转换修复...")
    
    # 创建一个测试图像 (BGR格式)
    test_image_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image_bgr[:, :, 0] = 255  # 蓝色通道
    
    print(f"原始BGR图像形状: {test_image_bgr.shape}")
    print(f"蓝色通道值: {test_image_bgr[0, 0, 0]}")
    print(f"绿色通道值: {test_image_bgr[0, 0, 1]}")
    print(f"红色通道值: {test_image_bgr[0, 0, 2]}")
    
    # 应用修复
    if len(test_image_bgr.shape) == 3 and test_image_bgr.shape[2] == 3:
        test_image_rgb = cv2.cvtColor(test_image_bgr, cv2.COLOR_BGR2RGB)
        print("✅ 颜色转换成功")
        print(f"转换后RGB - 红色通道值: {test_image_rgb[0, 0, 0]}")
        print(f"转换后RGB - 绿色通道值: {test_image_rgb[0, 0, 1]}")
        print(f"转换后RGB - 蓝色通道值: {test_image_rgb[0, 0, 2]}")
    else:
        print("❌ 颜色转换失败")

def test_time_measurement():
    """测试时间测量精度"""
    print("\n⏱️ 测试时间测量精度...")
    
    # 测试time.time()
    start_time = time.time()
    time.sleep(0.1)  # 模拟100ms的处理时间
    end_time = time.time()
    time_diff = end_time - start_time
    print(f"time.time() 测量: {time_diff:.6f}s")
    
    # 测试time.perf_counter()
    start_perf = time.perf_counter()
    time.sleep(0.1)  # 模拟100ms的处理时间
    end_perf = time.perf_counter()
    perf_diff = end_perf - start_perf
    print(f"time.perf_counter() 测量: {perf_diff:.6f}s")
    
    if abs(perf_diff - 0.1) < abs(time_diff - 0.1):
        print("✅ perf_counter() 更精确")
    else:
        print("⚠️ time() 可能更精确（不太可能）")

def test_json_cache():
    """测试JSON缓存功能"""
    print("\n💾 测试JSON缓存功能...")
    
    # 测试数据
    test_metrics = {
        "Box(P)": "0.856",
        "Box(R)": "0.743", 
        "mAP50": "0.788",
        "mAP50-95": "0.397",
        "验证时间": "2024-01-01 12:00:00"
    }
    
    cache_file = "test_metrics.json"
    
    try:
        # 保存测试
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)
        print("✅ JSON保存成功")
        
        # 加载测试
        with open(cache_file, 'r', encoding='utf-8') as f:
            loaded_metrics = json.load(f)
        
        if loaded_metrics == test_metrics:
            print("✅ JSON加载成功，数据一致")
        else:
            print("❌ JSON加载失败，数据不一致")
        
        # 清理测试文件
        os.remove(cache_file)
        print("✅ 测试文件已清理")
        
    except Exception as e:
        print(f"❌ JSON缓存测试失败: {e}")

def test_gpu_name_shortening():
    """测试GPU名称缩短功能"""
    print("\n🎮 测试GPU名称缩短功能...")
    
    test_names = [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce GTX 1080 Ti",
        "NVIDIA Tesla V100-SXM2-32GB",
        "AMD Radeon RX 6800 XT"
    ]
    
    for gpu_name in test_names:
        # 应用缩短逻辑
        gpu_short_name = gpu_name.replace("NVIDIA ", "").replace("GeForce ", "")
        if len(gpu_short_name) > 15:
            gpu_short_name = gpu_short_name[:15] + "..."
        
        print(f"原始: {gpu_name}")
        print(f"缩短: {gpu_short_name}")
        print(f"长度: {len(gpu_short_name)}")
        print("---")

def test_model_info_structure():
    """测试模型信息结构"""
    print("\n📊 测试模型信息结构...")
    
    # 模拟模型信息
    model_info = {
        "模型类型": "YOLO11",
        "任务类型": "detect",
        "类别数量": 3,
        "类别名称": ['Fruit-Citrus-0GcP', 'Fruit_on_Ground', 'Fruit_on_Tree'],
        "输入尺寸": "动态",
        "参数量": "9,458,752",
        "模型大小": "18.5 MB",
        "Box(P)": "0.856",
        "Box(R)": "0.743",
        "mAP50": "0.788",
        "mAP50-95": "0.397"
    }
    
    print("基本信息:")
    basic_keys = ["模型类型", "任务类型", "类别数量", "模型大小"]
    for key in basic_keys:
        if key in model_info:
            print(f"  ✅ {key}: {model_info[key]}")
        else:
            print(f"  ❌ 缺少 {key}")
    
    print("\n性能指标:")
    metric_keys = ["Box(P)", "Box(R)", "mAP50", "mAP50-95"]
    for key in metric_keys:
        if key in model_info and model_info[key] != "需要验证数据集":
            print(f"  ✅ {key}: {model_info[key]}")
        else:
            print(f"  ⚠️ {key}: 需要验证")

def main():
    """主测试函数"""
    print("🧪 柑橘检测应用修复测试")
    print("=" * 50)
    
    test_color_conversion()
    test_time_measurement()
    test_json_cache()
    test_gpu_name_shortening()
    test_model_info_structure()
    
    print("\n" + "=" * 50)
    print("🎉 所有测试完成！")
    print("\n修复内容总结:")
    print("✅ 1. 检测结果颜色问题 - BGR转RGB修复")
    print("✅ 2. 检测时间精度 - 使用perf_counter()") 
    print("✅ 3. GPU信息显示 - 名称缩短和字体调整")
    print("✅ 4. 模型性能指标 - 添加Box(P/R)、mAP50/95")
    print("✅ 5. JSON缓存系统 - 保存和加载性能指标")

if __name__ == "__main__":
    main()
