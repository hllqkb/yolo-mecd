# 基于YOLO11的柑橘检测系统

一个基于YOLO11架构的深度学习柑橘检测系统，集成增强注意力机制，实现精确的果实检测和分类。

## 项目概述

本项目实现了一个先进的柑橘检测系统，能够识别和分类不同状态的柑橘果实：树上果实、地面果实和一般柑橘果实。系统在验证数据集上达到了86.4% mAP50和48.2% mAP50-95的先进性能。

## 功能特性

- 高精度柑橘检测，mAP50达86.4%
- 多类别分类：树上果实、地面果实、一般柑橘
- 实时推理，单张图像161.3ms
- 基于Web的用户界面，交互便捷
- 批量处理功能
- 可自定义检测参数
- 性能监控和可视化

## 模型性能

### 验证结果
- 精确度(P): 83.9%
- 召回率(R): 79.1%
- mAP50: 86.4%
- mAP50-95: 48.2%
- 推理速度: 161.3ms/图像
- 预处理: 2.3ms/图像
- 后处理: 37.2ms/图像

### 数据集统计
- 验证图像: 119张
- 总实例数: 10,082个
- 类别数: 3类 (Fruit-Citrus-0GcP, Fruit_on_Ground, Fruit_on_Tree)

## 系统架构

系统基于YOLO11构建，包含以下增强功能：
- EMA（高效多尺度注意力）机制
- 优化的特征提取主干网络
- 增强的颈部架构，提升多尺度检测
- 改进的损失函数，提高收敛性

## 安装说明

### 系统要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU加速)
- Ultralytics YOLO
- Streamlit
- OpenCV
- PIL
- Plotly
- Pandas

### 安装步骤
```bash
# 克隆仓库
git clone https://github.com/yourusername/citrus-detection-yolo11.git
cd citrus-detection-yolo11

# 创建虚拟环境
python -m venv citrus_env
source citrus_env/bin/activate  # Windows: citrus_env\Scripts\activate

# 安装PyTorch（支持CUDA）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装Ultralytics和依赖
pip install ultralytics

# 安装Web界面依赖
pip install streamlit plotly psutil pandas opencv-python pillow

# 验证安装
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 快速开始
```bash
# 下载预训练模型（如果可用）
wget https://github.com/yourusername/citrus-detection-yolo11/releases/download/v1.0/best.pt

# 运行Web界面
streamlit run citrus_detection_app.py

# 或运行命令行推理
python -c "from ultralytics import YOLO; model = YOLO('best.pt'); results = model('test_image.jpg')"
```

## 使用方法

### Web界面
启动交互式Web应用：
```bash
streamlit run citrus_detection_app.py
```

在浏览器中访问 http://localhost:8501

### 命令行推理
```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/train15/weights/best.pt')

# 运行推理
results = model('path/to/your/image.jpg')

# 显示结果
results[0].show()
```

### 模型训练
从头开始训练模型：
```python
from ultralytics import YOLO

# 加载模型配置
model = YOLO('ultralytics/cfg/models/11/YOLO11_FINALLY.yaml')

# 训练模型
results = model.train(
    data='train/citdet.yaml',
    epochs=200,
    imgsz=1280,
    batch=4,
    device=0
)
```

## 数据集

模型在综合的柑橘检测数据集上训练，包含：
- 高分辨率柑橘果园图像（1280x1280像素）
- 多样化的光照条件和背景
- 多种柑橘品种和生长阶段
- 真实的农业环境
- 三个不同类别的标注：
  - Fruit-Citrus-0GcP: 一般柑橘果实（类别0）
  - Fruit_on_Ground: 地面掉落果实（类别1）
  - Fruit_on_Tree: 树上附着果实（类别2）

### 数据集统计
- 训练图像: 约1,000+张
- 验证图像: 119张
- 总标注实例: 10,082个
- 平均每张图像实例数: 84.7个
- 图像格式: JPG/PNG
- 标注格式: YOLO格式（.txt文件）

### 数据准备
数据集遵循YOLO格式结构：
```
datasets/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── citdet.yaml
```

### 数据增强
训练时应用的增强技术：
- 马赛克增强（4张图像组合）
- MixUp增强
- 随机水平翻转
- HSV色彩空间增强
- 随机缩放和平移
- Cutout增强

## 模型配置

关键训练参数：
- 图像尺寸: 1280x1280
- 批次大小: 4
- 训练轮数: 200
- 优化器: AdamW
- 学习率: 0.01（余弦退火）
- 数据增强: Mosaic, MixUp, HSV增强

## Web界面功能

### 检测模式
- 单张图像检测：上传并分析单张图像
- 批量处理：同时处理多张图像

### 可视化选项
- 可调节边界框粗细（1-5像素）
- 颜色编码类别识别：
  - 红色：一般柑橘果实
  - 绿色：地面果实
  - 蓝色：树上果实
- 置信度分数显示
- 标签可见性控制

### 性能监控
- 实时推理时间测量
- 系统资源监控（CPU、内存、GPU）
- 详细检测统计
- 交互式图表和可视化

## 结果分析

### 各类别性能
- Fruit_on_Ground: 地面果实检测精度高
- Fruit_on_Tree: 树上果实检测效果优秀
- 各类别性能均衡

### 速度优化
- 高效的预处理流水线
- 优化的推理引擎
- 快速后处理，适合实时应用

## 应用场景

### 农业监测
- 作物产量估算
- 收获计划和优化
- 质量评估和分级

### 研究应用
- 表型研究
- 生长模式分析
- 农业研究自动化数据收集

### 商业用途
- 自动分拣系统
- 加工设施质量控制
- 库存管理

## 技术细节

### 模型架构
基于YOLO11的自定义修改：
- 增强的CSPDarknet主干网络
- EMA（高效多尺度注意力）机制集成
- 优化的无锚检测头
- FPN/PAN多尺度特征融合
- 总参数量: 8,079,001
- 模型大小: 16.5 MB
- FLOPs: 24.4 GFLOPs

### 关键创新
1. **EMA注意力机制**: 在SPPF层后替换传统C2PSA注意力
2. **自定义YAML配置**: YOLO11_FINALLY.yaml优化架构
3. **多尺度训练**: 1280x1280输入分辨率实现高精度检测
4. **类别平衡损失**: 针对不平衡柑橘检测场景优化

### 训练策略
- 训练期间渐进式调整大小
- 先进的数据增强技术
- 精心调优的超参数
- 基于验证的早停
- 余弦退火学习率调度
- 预热轮数: 3
- 权重衰减: 0.0005
- 动量: 0.937

### 硬件要求
- 最低: 8GB GPU内存
- 推荐: 16GB+ GPU内存
- 训练时间: RTX 4070约4-6小时
- 推理: 现代GPU实时处理

### 性能基准
- CPU (Intel i7): 约800ms/图像
- GPU (RTX 4070): 约161ms/图像
- GPU (RTX 4090): 约120ms/图像
- 移动端（优化后）: 约300ms/图像

## 项目结构

```
citrus-detection-yolo11/
├── ultralytics/                    # 主项目目录
│   ├── cfg/models/11/              # 模型配置
│   │   └── YOLO11_FINALLY.yaml    # 自定义YOLO11架构
│   ├── nn/modules/                 # 神经网络模块
│   │   └── EMA.py                  # EMA注意力机制
│   ├── runs/detect/                # 训练和验证结果
│   │   ├── train15/                # 最新训练运行
│   │   │   ├── weights/            # 模型权重
│   │   │   │   ├── best.pt         # 最佳模型检查点
│   │   │   │   └── last.pt         # 最新模型检查点
│   │   │   ├── results.csv         # 训练指标
│   │   │   └── args.yaml           # 训练参数
│   │   └── val3/                   # 验证结果
│   ├── train/                      # 训练脚本和配置
│   │   ├── citdet.yaml             # 数据集配置
│   │   └── breakthrough_0.8.py    # 训练脚本
│   ├── datasets/                   # 数据集目录
│   │   └── dataverse_files/        # CitDet数据集
│   ├── citrus_detection_app.py     # Web应用程序
│   ├── requirements_app.txt        # Web应用依赖
│   ├── test_*.py                   # 测试脚本
│   └── README_CN.md                # 本文件
├── docs/                           # 文档
├── examples/                       # 使用示例
└── tests/                          # 单元测试
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少训练批次大小
   batch=2  # 而不是 batch=4

   # 使用更小的图像尺寸
   imgsz=640  # 而不是 imgsz=1280
   ```

2. **模型加载错误**
   ```python
   # 验证模型路径
   import os
   model_path = 'runs/detect/train15/weights/best.pt'
   print(f"模型存在: {os.path.exists(model_path)}")

   # 检查模型完整性
   from ultralytics import YOLO
   try:
       model = YOLO(model_path)
       print("模型加载成功")
   except Exception as e:
       print(f"错误: {e}")
   ```

3. **Web界面问题**
   ```bash
   # 检查Streamlit安装
   streamlit --version

   # 清除Streamlit缓存
   streamlit cache clear

   # 以调试模式运行
   streamlit run citrus_detection_app.py --logger.level=debug
   ```

4. **性能优化**
   ```python
   # 启用GPU加速
   import torch
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   model = YOLO('best.pt')
   results = model('image.jpg', device=device)

   # 多图像批量处理
   results = model(['img1.jpg', 'img2.jpg', 'img3.jpg'])
   ```

### 常见问题解答

**问：如何提高检测精度？**
答：尝试调整置信度阈值，使用更高分辨率图像，或在特定数据集上进行微调。

**问：能否用于其他水果检测？**
答：可以，但需要使用特定水果数据集重新训练模型并修改类别名称。

**问：如何导出模型用于移动端部署？**
```python
model = YOLO('best.pt')
model.export(format='onnx')  # 导出为ONNX
model.export(format='tflite')  # 导出为TensorFlow Lite
```

**问：如何在自定义数据集上训练？**
答：准备YOLO格式的数据集，更新citdet.yaml文件中的路径和类别名称，然后运行训练脚本。

## 论文引用

如果您在研究中使用此工作，请引用：

```
@article{citrus_detection_2024,
  title={基于增强注意力机制的YOLO11柑橘检测系统},
  author={您的姓名},
  journal={Agriculture},
  year={2024},
  publisher={MDPI}
}
```

相关工作：
- Agriculture期刊: https://www.mdpi.com/2073-4395/15/3/687

## 贡献指南

欢迎贡献！请随时提交Pull Request。对于重大更改，请先开issue讨论您想要更改的内容。

### 开发环境设置
1. Fork仓库
2. 创建功能分支
3. 进行更改
4. 添加测试（如适用）
5. 提交pull request

### 代码规范
- 遵循PEP 8 Python代码规范
- 使用有意义的变量名
- 为函数和类添加文档字符串
- 在适当的地方包含类型提示

## 许可证

本项目基于MIT许可证 - 详见LICENSE文件。

## 致谢

- Ultralytics的YOLO11架构
- 数据集贡献者和标注人员
- 开源社区提供的工具和库
- 农业研究社区的领域专业知识

## 联系方式

如有问题、建议或合作：
- 邮箱: your.email@example.com
- GitHub Issues: 使用issue跟踪器报告bug和功能请求
- 研究合作: 联系学术合作伙伴关系

## 更新日志

### 版本 1.1.0
- 增强的Web界面，支持批量处理
- 改进的可视化选项
- 性能监控仪表板
- 颜色编码类别识别

### 版本 1.0.0
- YOLO11实现的初始版本
- 基础Web界面
- 单图像检测功能
- 模型训练流水线

---

**注意**: 本项目仅用于研究和教育目的。商业使用请联系作者获取许可。
