#!/usr/bin/env python3
"""
柑橘检测Web应用启动脚本
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'opencv-python',
        'pillow',
        'ultralytics'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def main():
    """主函数"""
    print("🍊 柑橘智能检测系统启动器")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 获取应用路径
    app_path = Path(__file__).parent / "citrus_detection_app.py"
    
    if not app_path.exists():
        print(f"❌ 应用文件不存在: {app_path}")
        sys.exit(1)
    
    print(f"📁 应用路径: {app_path}")
    print("🚀 启动Streamlit应用...")
    print("=" * 50)
    
    # 启动Streamlit应用
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--theme.base", "light",
            "--theme.primaryColor", "#FF6B35",
            "--theme.backgroundColor", "#FFFFFF",
            "--theme.secondaryBackgroundColor", "#F0F2F6"
        ])
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
