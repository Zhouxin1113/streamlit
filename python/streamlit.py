import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # 分类任务用分类器
from sklearn.metrics import (
    accuracy_score, confusion_matrix, 
    classification_report
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 1. 页面配置（适配葡萄酒主题）
st.set_page_config(page_title="葡萄酒分类预测平台", layout="wide")
st.title("🍷 葡萄酒分类机器学习平台")
st.caption("基于 wine.csv 数据，预测葡萄酒类别（1/2/3类）")

# 2. 会话状态初始化（持久化模型/特征/均值）
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = []
if 'feature_means' not in st.session_state:
    st.session_state.feature_means = {}

# 3. 数据上传与加载（支持上传自定义wine.csv，也可默认加载示例）
st.sidebar.header("1. 数据加载")
uploaded_file = st.sidebar.file_uploader("上传 wine.csv 文件", type="csv")

# 处理数据加载（优先用上传文件，无上传则用默认路径）
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # 若用户未上传，尝试加载默认路径的wine.csv（Streamlit部署时需调整路径）
    try:
        df = pd.read_csv('/mnt/wine.csv')
        st.sidebar.success("已加载默认 wine.csv 数据")
    except:
        st.sidebar.error("请上传 wine.csv 文件或检查路径")

# 4. 数据预处理与展示（仅保留数值列，处理潜在缺失值）
if 'df' in locals():
    # 预处理：删除全空列，填充数值列缺失值（用均值）
    df = df.dropna(axis=1, how='all')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # 展示数据核心信息
    st.subheader("📊 数据概况")
    col1, col2, col3 = st.columns(3)
    col1.metric("数据行数", df.shape[0])
    col2.metric("特征列数", len(numeric_cols)-1)  # 排除标签列
    col3.metric("葡萄酒类别数", df['class'].nunique() if 'class' in df.columns else 0)

    # 数据预览（前10行）
    st.dataframe(df.head(10), use_container_width=True)

    # 5. 特征与标签选择（适配wine.csv，默认标签列为class）
    st.sidebar.header("2. 模型设置")
    # 特征列：默认选择除class外的所有数值列
    default_features = [col for col in numeric_cols if col != 'class']
    feature_cols = st.sidebar.multiselect(
        "选择特征列（默认：所有葡萄酒特征）",
        numeric_cols,
        default=default_features
    )
    # 标签列：默认选择class（葡萄酒类别）
    label_col = st.sidebar.selectbox(
        "选择标签列（预测目标）",
        numeric_cols,
        index=list(numeric_cols).index('class') if 'class' in numeric_cols else 0
    )

    # 更新会话状态
    st.session_state.feature_cols = feature_cols

    # 6. 模型训练（仅当特征和标签都选择后）
    if feature_cols and label_col and feature_cols != [label_col]:
        X = df[feature_cols]
        y = df[label_col]
        # 计算特征均值（用于预测默认值）
        st.session_state.feature_means = {col: float(X[col].mean()) for col in feature_cols}

        # 划分训练集/测试集（8:2）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y  # 分层抽样，保证类别分布
        )

        # 展示数据划分结果
        st.write(f"✅ 数据划分完成：训练集 {X_train.shape} | 测试集 {X_test.shape}")

        # 训练参数设置（决策树数量）
        n_estimators = st.sidebar.slider("决策树数量（随机森林）", 10, 200, 100)

        # 训练按钮
        if st.sidebar.button("🚀 开始训练模型"):
            with st.spinner("模型训练中...（约1-3秒）"):
                # 训练随机森林分类器
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=42,
                    max_depth=10  # 限制树深，避免过拟合
                )
                model.fit(X_train, y_train)
                # 保存模型到会话状态
                st.session_state.trained_model = model

                # 模型评估
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # 展示评估结果
                st.subheader("📈 模型评估结果")
                st.metric("测试集准确率", f"{accuracy:.4f}")  # 核心指标

                # 混淆矩阵可视化
                fig, ax = plt.subplots(figsize=(6, 4))
                cm = confusion_matrix(y_test, y_pred)
                im = ax.matshow(cm, cmap=plt.cm.Greens)
                plt.colorbar(im, ax=ax)
                ax.set_xlabel("Predicted Class", fontsize=10)
                ax.set_ylabel("True Class", fontsize=10)
                ax.set_title("Confusion Matrix（Predicted vs True）", fontsize=12)
                # 标注数值
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=12)
                st.pyplot(fig)

                # 详细分类报告（精确率、召回率、F1）
                st.subheader("📋 详细分类报告")
                report = classification_report(
                    y_test, y_pred, output_dict=True
                )
                # 转换为DataFrame展示
                report_df = pd.DataFrame(report).T.round(4)
                st.dataframe(report_df, use_container_width=True)

    else:
        st.warning("请选择**不同的特征列和标签列**（建议标签列为class）")

# 7. 在线预测模块（独立于训练流程，模型训练后永久可用）
st.subheader("🔍 葡萄酒类别在线预测")
if st.session_state.trained_model is None:
    st.info("请先完成「数据加载→模型训练」步骤后再预测")
else:
    # 输入特征值（默认填充训练集均值）
    st.caption("输入葡萄酒的特征值，点击预测类别（1/2/3）")
    input_data = {}
    for col in st.session_state.feature_cols:
        # 从会话状态获取均值作为默认值
        default_val = st.session_state.feature_means.get(col, 0.0)
        # 根据特征实际范围调整输入框（以经典wine数据为例）
        if col == 'alcohol':  # 酒精含量通常8-15
            input_data[col] = st.number_input(f"{col}（酒精含量）", value=default_val, min_value=8.0, max_value=15.0, step=0.1)
        elif col == 'malic_acid':  # 苹果酸通常0.7-5.8
            input_data[col] = st.number_input(f"{col}（苹果酸）", value=default_val, min_value=0.7, max_value=5.8, step=0.1)
        else:  # 其他特征用默认范围
            input_data[col] = st.number_input(f"{col}", value=default_val, min_value=0.0, step=0.01)

    # 预测按钮
    if st.button("✨ 开始预测"):
        try:
            # 转换输入为DataFrame（匹配模型输入格式）
            input_df = pd.DataFrame([input_data])
            # 预测
            pred_class = st.session_state.trained_model.predict(input_df)[0]
            # 展示结果
            st.success(f"🎉 预测结果：该葡萄酒属于 **{int(pred_class)}类**")
            
            # 展示预测概率（增加可信度）
            pred_proba = st.session_state.trained_model.predict_proba(input_df)[0]
            proba_df = pd.DataFrame({
                "葡萄酒类别": [1, 2, 3],
                "预测概率": [f"{p:.4f}" for p in pred_proba]
            })
            st.dataframe(proba_df, use_container_width=True)
        except Exception as e:
            st.error(f"预测出错：{str(e)}")
