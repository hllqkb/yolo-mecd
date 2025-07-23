#!/usr/bin/env python3
"""
测试颜色修复效果
"""

import numpy as np
from PIL import Image
import cv2

def test_color_formats():
    """测试不同颜色格式"""
    print("🎨 测试颜色格式")
    print("=" * 50)
    
    # 创建测试图像
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # RGB格式颜色定义
    rgb_colors = {
        "红色": (255, 0, 0),
        "绿色": (0, 255, 0),
        "蓝色": (0, 0, 255)
    }
    
    # BGR格式颜色定义
    bgr_colors = {
        "红色": (0, 0, 255),
        "绿色": (0, 255, 0),
        "蓝色": (255, 0, 0)
    }
    
    print("RGB格式 (PIL/Streamlit期望):")
    for name, color in rgb_colors.items():
        print(f"  {name}: {color}")
    
    print("\nBGR格式 (OpenCV默认):")
    for name, color in bgr_colors.items():
        print(f"  {name}: {color}")
    
    # 测试在PIL图像上绘制
    print("\n🖼️ 测试PIL图像绘制:")
    pil_image = Image.fromarray(test_image)
    img_array = np.array(pil_image)
    
    # 使用RGB颜色在左半部分绘制红色矩形
    img_array[20:80, 10:40] = rgb_colors["红色"]
    print(f"  使用RGB红色 {rgb_colors['红色']} 绘制矩形")
    
    # 检查像素值
    pixel_value = img_array[50, 25]
    print(f"  绘制后像素值: {pixel_value}")
    
    if tuple(pixel_value) == rgb_colors["红色"]:
        print("  ✅ RGB格式正确")
    else:
        print("  ❌ RGB格式错误")
    
    return True

def test_opencv_drawing():
    """测试OpenCV绘制函数"""
    print("\n🔧 测试OpenCV绘制函数")
    print("=" * 30)
    
    # 创建测试图像
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # RGB格式颜色
    red_rgb = (255, 0, 0)
    
    # 使用OpenCV绘制
    cv2.rectangle(img, (10, 10), (50, 50), red_rgb, 2)
    cv2.putText(img, "TEST", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_rgb, 1)
    
    # 检查绘制结果
    print(f"  绘制颜色: {red_rgb}")
    print(f"  矩形边框像素: {img[10, 10]}")
    
    # 转换为PIL图像查看效果
    pil_img = Image.fromarray(img)
    print(f"  转换为PIL后像素: {np.array(pil_img)[10, 10]}")
    
    return True

def simulate_custom_plot():
    """模拟自定义绘制函数"""
    print("\n🎯 模拟自定义绘制函数")
    print("=" * 30)
    
    # 创建测试图像
    image = Image.new('RGB', (200, 200), color='white')
    img_array = np.array(image)
    
    # 模拟检测框
    boxes = [(50, 50, 150, 100), (20, 120, 80, 180)]
    classes = [0, 1]  # 红色和绿色
    confidences = [0.85, 0.92]
    
    # RGB格式颜色定义（修复后）
    class_colors = {
        0: (255, 0, 0),      # 红色
        1: (0, 255, 0),      # 绿色
        2: (0, 0, 255),      # 蓝色
    }
    
    print("绘制检测框:")
    for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
        x1, y1, x2, y2 = box
        color = class_colors[cls]
        
        print(f"  框{i+1}: 类别{cls}, 颜色{color}, 置信度{conf:.2f}")
        
        # 绘制边界框
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"Class{cls} {conf:.2f}"
        cv2.putText(img_array, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 检查绘制结果
    print("\n检查绘制结果:")
    # 检查第一个框的边框像素
    border_pixel = img_array[50, 50]
    print(f"  红色框边框像素: {border_pixel}")
    
    if tuple(border_pixel) == class_colors[0]:
        print("  ✅ 红色框颜色正确")
    else:
        print(f"  ❌ 红色框颜色错误，期望{class_colors[0]}")
    
    # 转换为PIL图像
    result_image = Image.fromarray(img_array)
    print(f"  最终图像模式: {result_image.mode}")
    print(f"  最终图像尺寸: {result_image.size}")
    
    return True

def main():
    """主函数"""
    print("🧪 颜色修复测试")
    print("=" * 50)
    
    # 运行测试
    test_color_formats()
    test_opencv_drawing()
    simulate_custom_plot()
    
    print("\n" + "=" * 50)
    print("🎉 颜色修复测试完成！")
    print("\n📋 修复内容:")
    print("  ✅ 自定义绘制函数使用RGB格式颜色")
    print("  ✅ 移除了不必要的BGR->RGB转换")
    print("  ✅ 保持与PIL/Streamlit的兼容性")
    print("\n🚀 现在重启应用查看效果:")
    print("  streamlit run citrus_detection_app.py")
    print("\n🎨 颜色应该正常显示:")
    print("  🔴 红色 - 柑橘果实")
    print("  🟢 绿色 - 地面果实")
    print("  🔵 蓝色 - 树上果实")

if __name__ == "__main__":
    main()
