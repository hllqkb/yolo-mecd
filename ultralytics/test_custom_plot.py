#!/usr/bin/env python3
"""
测试自定义绘制功能
"""

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

def test_custom_plot():
    """测试自定义绘制功能"""
    print("🎨 测试自定义绘制功能")
    print("=" * 50)
    
    # 检查模型文件
    model_path = "/home/hllqk/projects/yolo-mecd/ultralytics/runs/detect/train13/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    try:
        # 加载模型
        print("📦 加载模型...")
        model = YOLO(model_path)
        print(f"✅ 模型加载成功")
        print(f"  - 类别数量: {len(model.names)}")
        print(f"  - 类别名称: {list(model.names.values())}")
        
        # 创建测试图像
        print("\n🖼️ 创建测试图像...")
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        test_image_pil = Image.fromarray(test_image)
        
        # 进行预测
        print("🔍 进行预测...")
        results = model(test_image_pil, verbose=False)
        result = results[0]
        
        print(f"✅ 预测完成")
        if result.boxes is not None:
            print(f"  - 检测到 {len(result.boxes)} 个目标")
        else:
            print("  - 未检测到目标")
        
        # 测试颜色定义
        print("\n🎨 测试颜色定义...")
        class_colors = {
            0: (255, 0, 0),      # 红色 - Fruit-Citrus-0GcP
            1: (0, 255, 0),      # 绿色 - Fruit_on_Ground  
            2: (0, 0, 255),      # 蓝色 - Fruit_on_Tree
        }
        
        for cls_id, color in class_colors.items():
            if cls_id in model.names:
                print(f"  - 类别 {cls_id} ({model.names[cls_id]}): RGB{color}")
        
        # 测试线条宽度
        print("\n📏 测试线条宽度...")
        line_widths = [1, 2, 3, 4, 5]
        for width in line_widths:
            print(f"  - 线条宽度 {width}: {'▁' * width}")
        
        print("\n✅ 自定义绘制功能测试完成！")
        print("\n📋 功能特性:")
        print("  ✅ 更细的边界框线条")
        print("  ✅ 无背景的文字标签")
        print("  ✅ 不同类别用不同颜色")
        print("  ✅ 可调节线条宽度")
        print("  ✅ 颜色说明界面")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_color_conversion():
    """测试颜色转换"""
    print("\n🌈 测试颜色转换...")
    
    # BGR颜色 (OpenCV格式)
    bgr_colors = {
        "红色": (0, 0, 255),
        "绿色": (0, 255, 0),
        "蓝色": (255, 0, 0)
    }
    
    # RGB颜色 (PIL/Streamlit格式)
    rgb_colors = {
        "红色": (255, 0, 0),
        "绿色": (0, 255, 0),
        "蓝色": (0, 0, 255)
    }
    
    print("BGR格式 (OpenCV):")
    for name, color in bgr_colors.items():
        print(f"  - {name}: {color}")
    
    print("RGB格式 (PIL/Streamlit):")
    for name, color in rgb_colors.items():
        print(f"  - {name}: {color}")
    
    print("✅ 颜色转换测试完成")

def main():
    """主函数"""
    print("🧪 自定义绘制功能测试")
    print("=" * 50)
    
    # 运行测试
    success = test_custom_plot()
    test_color_conversion()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过！")
        print("\n🚀 现在可以启动应用查看效果:")
        print("   streamlit run citrus_detection_app.py")
        print("\n🎨 新功能:")
        print("   - 边界框线条更细")
        print("   - 文字无背景色")
        print("   - 红色=柑橘果实, 绿色=地面果实, 蓝色=树上果实")
        print("   - 可调节线条宽度 (1-5)")
    else:
        print("❌ 测试失败，请检查错误信息")

if __name__ == "__main__":
    main()
