#!/bin/bash

# 柑橘检测Web应用启动脚本

echo "🍊 柑橘智能检测系统启动器"
echo "=================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装或不在PATH中"
    exit 1
fi

echo "✅ Python环境检查通过"

# 检查必要的包
echo "🔍 检查依赖包..."

# 尝试安装缺失的包
missing_packages=()

python -c "import streamlit" 2>/dev/null || missing_packages+=("streamlit")
python -c "import plotly" 2>/dev/null || missing_packages+=("plotly")
python -c "import psutil" 2>/dev/null || missing_packages+=("psutil")

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo "📦 安装缺失的依赖包: ${missing_packages[*]}"
    pip install "${missing_packages[@]}"
    
    if [ $? -ne 0 ]; then
        echo "❌ 依赖包安装失败"
        exit 1
    fi
fi

echo "✅ 所有依赖包已就绪"

# 检查模型文件
model_path="/home/hllqk/projects/yolo-mecd/ultralytics/runs/detect/train13/weights/best.pt"
if [ -f "$model_path" ]; then
    echo "✅ 默认模型文件存在"
else
    echo "⚠️  默认模型文件不存在，可以在应用中上传自定义模型"
fi

# 启动应用
echo "🚀 启动Streamlit应用..."
echo "=================================="

# 设置环境变量
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 启动应用
streamlit run citrus_detection_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --theme.base light \
    --theme.primaryColor "#FF6B35" \
    --theme.backgroundColor "#FFFFFF" \
    --theme.secondaryBackgroundColor "#F0F2F6" \
    --browser.gatherUsageStats false

echo "👋 应用已停止"
