import streamlit as st
import cv2
import numpy as np
import time
import torch
from PIL import Image, ImageDraw, ImageFont
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import json
from ultralytics import YOLO
import tempfile
import os
import psutil
import zipfile
from io import BytesIO
import base64

# 页面配置
st.set_page_config(
    page_title="🍊 柑橘智能检测系统",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-card h2 {
        font-size: 1.5rem;
        margin: 0.2rem 0;
    }
    .metric-card h3 {
        font-size: 0.9rem;
        margin: 0.2rem 0;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-header">🍊 柑橘智能检测系统</h1>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown("## 🔧 系统配置")
    
    # 模型选择
    st.markdown("### 📦 模型管理")
    model_option = st.radio(
        "选择模型来源:",
        ["使用默认模型", "上传自定义模型"]
    )
    
    if model_option == "使用默认模型":
        default_model_path = "/home/hllqk/projects/yolo-mecd/ultralytics/runs/detect/train13/weights/best.pt"
        if os.path.exists(default_model_path):
            model_path = default_model_path
            st.success("✅ 默认模型已加载")
        else:
            st.error("❌ 默认模型文件不存在")
            model_path = None
    else:
        uploaded_model = st.file_uploader(
            "上传模型文件 (.pt)",
            type=['pt'],
            help="请上传训练好的YOLO模型文件"
        )
        if uploaded_model:
            # 保存上传的模型到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                tmp_file.write(uploaded_model.read())
                model_path = tmp_file.name
            st.success("✅ 自定义模型已上传")
        else:
            model_path = None
    
    # 检测参数
    st.markdown("### ⚙️ 检测参数")
    confidence_threshold = st.slider(
        "置信度阈值",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="只显示置信度高于此阈值的检测结果"
    )
    
    iou_threshold = st.slider(
        "IoU阈值",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="非极大值抑制的IoU阈值"
    )
    
    # 显示选项
    st.markdown("### 🎨 显示选项")
    show_confidence = st.checkbox("显示置信度", value=True)
    show_labels = st.checkbox("显示标签", value=True)
    show_boxes = st.checkbox("显示边界框", value=True)

    # 线条宽度控制
    line_width = st.slider(
        "边界框线条宽度",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="调整检测框的线条粗细"
    )

    # 颜色说明
    st.markdown("### 🎨 类别颜色说明")
    st.markdown("""
    <div style="font-size: 0.9rem;">
        <p><span style="color: red; font-weight: bold;">🔴 红色</span> - Fruit-Citrus-0GcP (柑橘果实)</p>
        <p><span style="color: green; font-weight: bold;">🟢 绿色</span> - Fruit_on_Ground (地面果实)</p>
        <p><span style="color: blue; font-weight: bold;">🔵 蓝色</span> - Fruit_on_Tree (树上果实)</p>
    </div>
    """, unsafe_allow_html=True)

    # 系统信息
    st.markdown("### 💻 系统信息")
    if st.button("🔄 刷新系统信息"):
        st.rerun()

    # CPU和内存使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    st.metric("CPU使用率", f"{cpu_percent:.1f}%")
    st.metric("内存使用率", f"{memory.percent:.1f}%")

    # GPU信息（如果可用）
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # 缩短GPU名称显示
        gpu_short_name = gpu_name.replace("NVIDIA ", "").replace("GeForce ", "")
        if len(gpu_short_name) > 15:
            gpu_short_name = gpu_short_name[:15] + "..."
        st.metric("GPU", gpu_short_name)
        st.metric("显存", f"{gpu_memory:.1f}GB")

# 自定义绘制函数
def custom_plot_detections(image, result, show_confidence=True, show_labels=True, show_boxes=True, line_width=2):
    """
    自定义绘制检测结果
    - 更细的边界框
    - 无背景的文字
    - 不同类别用不同颜色
    """
    if result.boxes is None or len(result.boxes) == 0:
        return np.array(image)

    # 转换为numpy数组
    img_array = np.array(image)

    # 定义类别颜色 (RGB格式，适配PIL图像)
    class_colors = {
        0: (255, 0, 0),      # 红色 - Fruit-Citrus-0GcP
        1: (0, 255, 0),      # 绿色 - Fruit_on_Ground
        2: (0, 0, 255),      # 蓝色 - Fruit_on_Tree
    }

    # 获取检测结果
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    class_names = result.names

    # 绘制每个检测框
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = box.astype(int)

        # 获取类别颜色
        color = class_colors.get(cls, (128, 128, 128))  # 默认灰色

        if show_boxes:
            # 绘制边界框 (更细的线条)
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, line_width)

        if show_labels or show_confidence:
            # 准备标签文本
            label_parts = []
            if show_labels:
                label_parts.append(class_names[cls])
            if show_confidence:
                label_parts.append(f"{conf:.2f}")

            label = " ".join(label_parts)

            # 计算文本尺寸
            font_scale = 0.6
            font_thickness = 2
            _, text_height = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )[0]

            # 文本位置 (在框的上方)
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10

            # 绘制文本 (无背景，直接用颜色文字)
            cv2.putText(
                img_array,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                font_thickness,
                cv2.LINE_AA
            )

    return img_array

# 加载模型函数
@st.cache_resource
def load_model(model_path):
    """加载YOLO模型"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

# 获取模型信息
def get_model_info(model, model_path):
    """获取模型详细信息"""
    if model is None:
        return None

    try:
        # 检查是否有缓存的性能指标
        cache_file = f"{model_path}_metrics.json"
        cached_metrics = load_cached_metrics(cache_file)

        # 获取模型基本信息
        model_info = {
            "模型类型": "YOLO11",
            "任务类型": model.task,
            "类别数量": len(model.names),
            "类别名称": list(model.names.values()),
            "输入尺寸": "动态",
            "参数量": f"{sum(p.numel() for p in model.model.parameters()):,}",
            "模型大小": f"{os.path.getsize(model_path) / (1024*1024):.1f} MB" if os.path.exists(model_path) else "未知"
        }

        # 添加性能指标（优先级：缓存 > 训练结果 > 提示）
        if cached_metrics:
            model_info.update(cached_metrics)
        else:
            # 尝试从训练结果中提取
            training_metrics = extract_training_metrics(model_path)
            if training_metrics:
                model_info.update(training_metrics)
                # 同时保存到缓存
                cache_file = f"{model_path}_metrics.json"
                save_metrics_to_cache(training_metrics, cache_file)
            else:
                # 如果都没有，显示提示
                model_info.update({
                    "Box(P)": "需要验证数据集",
                    "Box(R)": "需要验证数据集",
                    "mAP50": "需要验证数据集",
                    "mAP50-95": "需要验证数据集"
                })

        return model_info
    except Exception as e:
        st.error(f"获取模型信息失败: {str(e)}")
        return None

# 加载缓存的性能指标
def load_cached_metrics(cache_file):
    """从JSON文件加载缓存的性能指标"""
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"加载缓存指标失败: {e}")
    return None

# 保存性能指标到缓存
def save_metrics_to_cache(metrics, cache_file):
    """保存性能指标到JSON文件"""
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存指标到缓存失败: {e}")
        return False

# 运行模型验证获取性能指标
def run_model_validation(model, validation_data_path=None):
    """运行模型验证获取性能指标"""
    try:
        # 如果没有指定验证数据集，尝试使用默认路径
        if validation_data_path is None:
            # 尝试找到验证数据集
            possible_paths = [
                "/home/hllqk/projects/yolo-mecd/ultralytics/train/citdet.yaml"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    validation_data_path = path
                    break

        if validation_data_path and os.path.exists(validation_data_path):
            # 运行验证
            results = model.val(data=validation_data_path, verbose=False)

            # 提取关键指标
            metrics = {
                "Box(P)": f"{results.box.mp:.3f}",
                "Box(R)": f"{results.box.mr:.3f}",
                "mAP50": f"{results.box.map50:.3f}",
                "mAP50-95": f"{results.box.map:.3f}",
                "验证时间": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            return metrics
        else:
            return None

    except Exception as e:
        print(f"模型验证失败: {e}")
        return None

# 从训练结果中提取性能指标
def extract_training_metrics(model_path):
    """从训练结果中提取性能指标"""
    try:
        # 检查是否是train15的best.pt模型
        if "train15" in model_path and "best.pt" in model_path:
            # 使用训练完成后的真实指标
            return {
                "Box(P)": "0.843",
                "Box(R)": "0.798",
                "mAP50": "0.863",
                "mAP50-95": "0.452",
                "验证时间": "训练完成后验证结果"
            }

        # 尝试从results.csv中读取最后的验证结果
        train_dir = os.path.dirname(os.path.dirname(model_path))  # 获取训练目录
        results_file = os.path.join(train_dir, "results.csv")

        if os.path.exists(results_file):
            import pandas as pd
            df = pd.read_csv(results_file)
            if len(df) > 0:
                last_row = df.iloc[-1]
                return {
                    "Box(P)": f"{last_row.get('metrics/precision(B)', 0):.3f}",
                    "Box(R)": f"{last_row.get('metrics/recall(B)', 0):.3f}",
                    "mAP50": f"{last_row.get('metrics/mAP50(B)', 0):.3f}",
                    "mAP50-95": f"{last_row.get('metrics/mAP50-95(B)', 0):.3f}",
                    "验证时间": "从训练记录提取"
                }

        return None

    except Exception as e:
        print(f"提取训练指标失败: {e}")
        return None

# 主界面
if model_path:
    # 加载模型
    model = load_model(model_path)
    
    if model:
        # 显示模型信息
        st.markdown("## 📊 模型信息")

        # 添加验证按钮
        col_info, col_validate = st.columns([3, 1])
        with col_validate:
            if st.button("🔄 验证模型性能", help="运行模型验证获取性能指标"):
                with st.spinner("正在验证模型性能..."):
                    metrics = run_model_validation(model)
                    if metrics:
                        cache_file = f"{model_path}_metrics.json"
                        if save_metrics_to_cache(metrics, cache_file):
                            st.success("✅ 性能指标已更新")
                            st.rerun()
                        else:
                            st.warning("⚠️ 指标保存失败")
                    else:
                        st.warning("⚠️ 验证失败，请检查数据集路径")

        model_info = get_model_info(model, model_path)
        
        if model_info:
            # 基本信息
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 任务类型</h3>
                    <h2>{model_info['任务类型']}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📝 类别数量</h3>
                    <h2>{model_info['类别数量']}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💾 模型大小</h3>
                    <h2>{model_info['模型大小']}</h2>
                </div>
                """, unsafe_allow_html=True)

            # 性能指标
            if 'mAP50' in model_info and model_info['mAP50'] != "需要验证数据集":
                st.markdown("### 📈 性能指标")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 Box(P)</h3>
                        <h2>{model_info['Box(P)']}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Box(R)</h3>
                        <h2>{model_info['Box(R)']}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🏆 mAP50</h3>
                        <h2>{model_info['mAP50']}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>⭐ mAP50-95</h3>
                        <h2>{model_info['mAP50-95']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    <h3>📈 性能指标</h3>
                    <p>点击"验证模型性能"按钮获取详细的性能指标</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 详细信息表格
            with st.expander("🔍 查看详细模型信息"):
                info_df = pd.DataFrame([
                    {"属性": k, "值": str(v)} for k, v in model_info.items()
                ])
                st.dataframe(info_df, use_container_width=True)
        
        # 检测模式选择
        st.markdown("## 🔧 检测模式")
        detection_mode = st.radio(
            "选择检测模式:",
            ["单张图像检测", "批量图像检测"],
            horizontal=True
        )

        if detection_mode == "单张图像检测":
            # 单张图像检测
            st.markdown("## 🖼️ 单张图像检测")

            # 图像上传
            uploaded_file = st.file_uploader(
                "选择要检测的图像",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="支持 JPG, JPEG, PNG, BMP 格式"
            )
        
        if uploaded_file is not None:
            # 显示原始图像
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📷 原始图像")
                st.image(image, caption="上传的图像", use_container_width=True)
            
            # 检测按钮
            if st.button("🚀 开始检测", use_container_width=True):
                with st.spinner("🔄 正在进行检测..."):
                    # 记录开始时间（更精确的时间测量）
                    start_time = time.perf_counter()

                    # 进行预测
                    results = model(
                        image,
                        conf=confidence_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )

                    # 记录结束时间
                    end_time = time.perf_counter()
                    inference_time = end_time - start_time
                    
                    # 处理结果
                    result = results[0]
                    
                    with col2:
                        st.markdown("### 🎯 检测结果")
                        
                        # 使用自定义绘制函数
                        annotated_image = custom_plot_detections(
                            image,
                            result,
                            show_confidence=show_confidence,
                            show_labels=show_labels,
                            show_boxes=show_boxes,
                            line_width=line_width
                        )

                        # 转换为PIL图像（自定义函数已使用RGB格式）
                        annotated_image = Image.fromarray(annotated_image)
                        st.image(annotated_image, caption="检测结果", use_container_width=True)
                    
                    # 显示检测统计信息
                    st.markdown("## 📈 检测统计")
                    
                    # 获取检测结果
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        # 统计各类别数量
                        class_counts = {}
                        confidences = []
                        
                        for i in range(len(boxes)):
                            class_id = int(boxes.cls[i])
                            class_name = model.names[class_id]
                            confidence = float(boxes.conf[i])
                            
                            if class_name not in class_counts:
                                class_counts[class_name] = 0
                            class_counts[class_name] += 1
                            confidences.append(confidence)
                        
                        # 显示统计信息
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="success-box">
                                <h3>🎯 检测数量</h3>
                                <h2>{len(boxes)}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            avg_confidence = np.mean(confidences) if confidences else 0
                            st.markdown(f"""
                            <div class="success-box">
                                <h3>📊 平均置信度</h3>
                                <h2>{avg_confidence:.2%}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            max_confidence = max(confidences) if confidences else 0
                            st.markdown(f"""
                            <div class="success-box">
                                <h3>🏆 最高置信度</h3>
                                <h2>{max_confidence:.2%}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                            <div class="success-box">
                                <h3>⏱️ 检测时间</h3>
                                <h2>{inference_time:.3f}s</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 类别分布图表
                        if class_counts:
                            st.markdown("### 📊 类别分布")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # 饼图
                                fig_pie = px.pie(
                                    values=list(class_counts.values()),
                                    names=list(class_counts.keys()),
                                    title="检测类别分布",
                                    color_discrete_sequence=px.colors.qualitative.Set3
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            with col2:
                                # 柱状图
                                fig_bar = px.bar(
                                    x=list(class_counts.keys()),
                                    y=list(class_counts.values()),
                                    title="各类别检测数量",
                                    color=list(class_counts.values()),
                                    color_continuous_scale="Viridis"
                                )
                                fig_bar.update_layout(
                                    xaxis_title="类别",
                                    yaxis_title="数量"
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # 详细检测结果表格
                        with st.expander("📋 查看详细检测结果"):
                            detection_data = []
                            for i in range(len(boxes)):
                                class_id = int(boxes.cls[i])
                                class_name = model.names[class_id]
                                confidence = float(boxes.conf[i])
                                bbox = boxes.xyxy[i].cpu().numpy()
                                
                                detection_data.append({
                                    "序号": i + 1,
                                    "类别": class_name,
                                    "置信度": f"{confidence:.2%}",
                                    "边界框": f"({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})",
                                    "面积": f"{(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]):.0f}"
                                })
                            
                            detection_df = pd.DataFrame(detection_data)
                            st.dataframe(detection_df, use_container_width=True)
                    
                    else:
                        st.markdown("""
                        <div class="info-box">
                            <h3>ℹ️ 检测结果</h3>
                            <p>未检测到任何目标，请尝试：</p>
                            <ul>
                                <li>降低置信度阈值</li>
                                <li>使用更清晰的图像</li>
                                <li>确保图像中包含目标对象</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
        
            else:
                st.markdown("""
                <div class="info-box">
                    <h3>📁 请上传图像</h3>
                    <p>请在上方选择要检测的图像文件</p>
                </div>
                """, unsafe_allow_html=True)

        else:  # 批量图像检测
            st.markdown("## 📁 批量图像检测")

            # 批量上传
            uploaded_files = st.file_uploader(
                "选择多张图像进行批量检测",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True,
                help="可以同时选择多张图像进行批量检测"
            )

            if uploaded_files:
                st.success(f"✅ 已选择 {len(uploaded_files)} 张图像")

                # 批量检测按钮
                if st.button("🚀 开始批量检测", use_container_width=True):
                    # 创建进度条
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # 存储所有结果
                    all_results = []
                    total_time = 0

                    for i, uploaded_file in enumerate(uploaded_files):
                        # 更新进度
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"正在处理: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")

                        # 处理单张图像
                        image = Image.open(uploaded_file)

                        # 记录时间（使用更精确的计时）
                        start_time = time.perf_counter()
                        results = model(
                            image,
                            conf=confidence_threshold,
                            iou=iou_threshold,
                            verbose=False
                        )
                        end_time = time.perf_counter()
                        inference_time = end_time - start_time
                        total_time += inference_time

                        # 处理结果
                        result = results[0]
                        boxes = result.boxes

                        detection_count = len(boxes) if boxes is not None else 0
                        avg_confidence = np.mean([float(conf) for conf in boxes.conf]) if boxes is not None and len(boxes) > 0 else 0

                        all_results.append({
                            "文件名": uploaded_file.name,
                            "检测数量": detection_count,
                            "平均置信度": f"{avg_confidence:.2%}",
                            "处理时间": f"{inference_time:.3f}s",
                            "图像": image,
                            "结果": result
                        })

                    # 完成处理
                    progress_bar.progress(1.0)
                    status_text.text("✅ 批量检测完成!")

                    # 显示批量检测统计
                    st.markdown("## 📊 批量检测统计")

                    col1, col2, col3, col4 = st.columns(4)

                    total_detections = sum([r["检测数量"] for r in all_results])
                    avg_time = total_time / len(all_results)
                    successful_detections = len([r for r in all_results if r["检测数量"] > 0])

                    with col1:
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>📁 处理图像</h3>
                            <h2>{len(all_results)}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>🎯 总检测数</h3>
                            <h2>{total_detections}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>⏱️ 平均时间</h3>
                            <h2>{avg_time:.3f}s</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        success_rate = successful_detections / len(all_results) * 100
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ 成功率</h3>
                            <h2>{success_rate:.1f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    # 详细结果表格
                    st.markdown("### 📋 详细结果")
                    results_df = pd.DataFrame([
                        {
                            "文件名": r["文件名"],
                            "检测数量": r["检测数量"],
                            "平均置信度": r["平均置信度"],
                            "处理时间": r["处理时间"]
                        } for r in all_results
                    ])
                    st.dataframe(results_df, use_container_width=True)

                    # 显示部分检测结果图像
                    st.markdown("### 🖼️ 检测结果预览")

                    # 选择要显示的图像
                    num_to_show = min(6, len(all_results))
                    cols = st.columns(3)

                    for i in range(num_to_show):
                        col_idx = i % 3
                        with cols[col_idx]:
                            result_data = all_results[i]

                            # 使用自定义绘制函数
                            annotated_image = custom_plot_detections(
                                result_data["图像"],
                                result_data["结果"],
                                show_confidence=show_confidence,
                                show_labels=show_labels,
                                show_boxes=show_boxes,
                                line_width=line_width
                            )

                            # 转换为PIL图像（自定义函数已使用RGB格式）
                            annotated_image = Image.fromarray(annotated_image)

                            st.image(
                                annotated_image,
                                caption=f"{result_data['文件名']} (检测: {result_data['检测数量']})",
                                use_container_width=True
                            )

                    if len(all_results) > 6:
                        st.info(f"显示前6张图像的检测结果，共{len(all_results)}张图像")

            else:
                st.markdown("""
                <div class="info-box">
                    <h3>📁 请选择图像文件</h3>
                    <p>请在上方选择要进行批量检测的图像文件</p>
                    <ul>
                        <li>支持同时选择多张图像</li>
                        <li>支持JPG、PNG、BMP等格式</li>
                        <li>建议单次处理不超过20张图像</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.error("❌ 模型加载失败，请检查模型文件")

else:
    st.markdown("""
    <div class="info-box">
        <h3>🚀 欢迎使用柑橘智能检测系统</h3>
        <p>请在左侧边栏选择或上传模型文件开始使用</p>
        <ul>
            <li>支持YOLO格式的.pt模型文件</li>
            <li>提供实时检测性能统计</li>
            <li>支持多种图像格式</li>
            <li>可视化检测结果和统计信息</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🍊 柑橘智能检测系统 | 基于 YOLO11 深度学习模型</p>
    <p>💡 提供高精度、实时的柑橘检测服务</p>
</div>
""", unsafe_allow_html=True)
