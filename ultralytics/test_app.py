#!/usr/bin/env python3
"""
柑橘检测Web应用测试脚本
"""

import sys
import os
from pathlib import Path

def test_imports():
    """测试必要的包导入"""
    print("🔍 测试包导入...")
    
    try:
        import streamlit as st
        print("✅ Streamlit")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        return False
    
    try:
        import plotly.express as px
        print("✅ Plotly")
    except ImportError as e:
        print(f"❌ Plotly: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas")
    except ImportError as e:
        print(f"❌ Pandas: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics")
    except ImportError as e:
        print(f"❌ Ultralytics: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL")
    except ImportError as e:
        print(f"❌ PIL: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch")
        if torch.cuda.is_available():
            print(f"  🎮 GPU可用: {torch.cuda.get_device_name(0)}")
        else:
            print("  💻 仅CPU模式")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    return True

def test_model_loading():
    """测试模型加载"""
    print("\n🤖 测试模型加载...")
    
    try:
        from ultralytics import YOLO
        
        # 测试默认模型路径
        default_model_path = "/home/hllqk/projects/yolo-mecd/ultralytics/runs/detect/train13/weights/best.pt"
        
        if os.path.exists(default_model_path):
            print(f"✅ 默认模型文件存在: {default_model_path}")
            
            try:
                model = YOLO(default_model_path)
                print(f"✅ 模型加载成功")
                print(f"  📊 任务类型: {model.task}")
                print(f"  🏷️  类别数量: {len(model.names)}")
                print(f"  📝 类别名称: {list(model.names.values())}")
                return True
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                return False
        else:
            print(f"⚠️  默认模型文件不存在: {default_model_path}")
            print("   可以在应用中上传自定义模型")
            return True
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

def test_app_file():
    """测试应用文件"""
    print("\n📁 测试应用文件...")
    
    app_file = Path("citrus_detection_app.py")
    
    if app_file.exists():
        print(f"✅ 应用文件存在: {app_file}")
        
        # 检查文件大小
        file_size = app_file.stat().st_size
        print(f"  📏 文件大小: {file_size} bytes")
        
        # 简单语法检查
        try:
            with open(app_file, 'r', encoding='utf-8') as f:
                content = f.read()
                compile(content, app_file, 'exec')
            print("✅ 语法检查通过")
            return True
        except SyntaxError as e:
            print(f"❌ 语法错误: {e}")
            return False
    else:
        print(f"❌ 应用文件不存在: {app_file}")
        return False

def test_system_resources():
    """测试系统资源"""
    print("\n💻 测试系统资源...")
    
    try:
        import psutil
        
        # CPU信息
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"✅ CPU: {cpu_count}核, 使用率: {cpu_percent:.1f}%")
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"✅ 内存: {memory_gb:.1f}GB, 使用率: {memory.percent:.1f}%")
        
        # 磁盘空间
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        print(f"✅ 磁盘可用空间: {disk_free_gb:.1f}GB")
        
        return True
        
    except ImportError:
        print("⚠️  psutil未安装，跳过系统资源检查")
        return True
    except Exception as e:
        print(f"❌ 系统资源检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🍊 柑橘检测Web应用测试")
    print("=" * 50)
    
    tests = [
        ("包导入测试", test_imports),
        ("模型加载测试", test_model_loading),
        ("应用文件测试", test_app_file),
        ("系统资源测试", test_system_resources)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！应用可以正常启动")
        print("\n🚀 启动应用:")
        print("   方法1: python run_app.py")
        print("   方法2: streamlit run citrus_detection_app.py")
        print("   方法3: ./start_app.sh")
    else:
        print("⚠️  部分测试失败，请检查相关问题")
        
        # 提供安装建议
        print("\n💡 安装建议:")
        print("   pip install streamlit plotly psutil")
        print("   pip install ultralytics opencv-python pillow")

if __name__ == "__main__":
    main()
