# Citrus Detection System - YOLO-MECD

## Language Selection / 语言选择

Choose your preferred language for documentation:

### English
- [English README](PROJECT_README.md) - Complete documentation in English
- Features: Installation, usage, technical details, troubleshooting

### 中文 (Chinese)
- [中文说明文档](README_CN.md) - 完整的中文文档
- 包含：安装说明、使用方法、技术细节、故障排除

---

## Quick Links / 快速链接

### For Developers / 开发者
- [Installation Guide](PROJECT_README.md#installation) / [安装指南](README_CN.md#安装说明)
- [Usage Examples](PROJECT_README.md#usage) / [使用示例](README_CN.md#使用方法)
- [API Documentation](PROJECT_README.md#technical-details) / [API文档](README_CN.md#技术细节)

### For Researchers / 研究人员
- [Model Performance](PROJECT_README.md#model-performance) / [模型性能](README_CN.md#模型性能)
- [Dataset Information](PROJECT_README.md#dataset) / [数据集信息](README_CN.md#数据集)
- [Citation](PROJECT_README.md#citation) / [论文引用](README_CN.md#论文引用)

### For Users / 用户
- [Web Interface Guide](PROJECT_README.md#web-interface) / [Web界面指南](README_CN.md#web界面功能)
- [Troubleshooting](PROJECT_README.md#troubleshooting) / [故障排除](README_CN.md#故障排除)
- [FAQ](PROJECT_README.md#faq) / [常见问题](README_CN.md#常见问题解答)

---

## Project Overview / 项目概述

**English**: A deep learning-based citrus detection system using YOLO11 architecture with enhanced attention mechanisms for accurate fruit detection and classification.

**中文**: 一个基于YOLO11架构的深度学习柑橘检测系统，集成增强注意力机制，实现精确的果实检测和分类。

### Key Features / 主要特性

- **High Accuracy / 高精度**: 86.4% mAP50
- **Real-time Processing / 实时处理**: 161.3ms per image / 每张图像161.3ms
- **Multi-class Detection / 多类别检测**: Tree fruits, ground fruits, general citrus / 树上果实、地面果实、一般柑橘
- **Web Interface / Web界面**: User-friendly interface / 用户友好界面
- **Batch Processing / 批量处理**: Multiple images simultaneously / 同时处理多张图像

---

## Quick Start / 快速开始

### Installation / 安装
```bash
# Clone repository / 克隆仓库
git clone https://github.com/yourusername/citrus-detection-yolo11.git
cd citrus-detection-yolo11

# Install dependencies / 安装依赖
pip install -r requirements.txt

# Run web interface / 运行Web界面
streamlit run citrus_detection_app.py
```

### Basic Usage / 基本使用
```python
from ultralytics import YOLO

# Load model / 加载模型
model = YOLO('runs/detect/train15/weights/best.pt')

# Run detection / 运行检测
results = model('your_image.jpg')

# Show results / 显示结果
results[0].show()
```

---

## Support / 技术支持

- **GitHub Issues**: Report bugs and request features / 报告bug和功能请求
- **Email**: your.email@example.com
- **Documentation**: Comprehensive guides in both languages / 两种语言的完整指南

---

## License / 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

本项目基于MIT许可证 - 详见[LICENSE](LICENSE)文件。
