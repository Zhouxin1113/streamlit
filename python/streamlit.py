import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# 1. 页面配置
st.set_page_config(page_title="ML Platform by Streamlit", layout="wide")
st.title("Streamlit 机器学习可视化平台")

# 2. 数据上传模块
st.sidebar.header("1. 数据上传")
uploaded_file = st.sidebar.file_uploader("上传CSV文件", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("数据预览")
    st.dataframe(df.head(10))

    # 3. 数据预处理（选择特征和标签）
    st.sidebar.header("2. 预处理设置")
    feature_cols = st.sidebar.multiselect("选择特征列", df.columns)
    label_col = st.sidebar.selectbox("选择标签列", df.columns)

    if feature_cols and label_col:
        X = df[feature_cols]
        y = df[label_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"训练集大小: {X_train.shape} | 测试集大小: {X_test.shape}")

        # 4. 模型训练模块
        st.sidebar.header("3. 模型训练")
        n_estimators = st.sidebar.slider("决策树数量", 10, 200, 100)
        if st.sidebar.button("开始训练"):
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.subheader("模型评估结果")
            st.write(f"**准确率**: {accuracy:.4f}")

            # 5. 结果可视化
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            ax.matshow(cm, cmap=plt.cm.Blues)
            st.pyplot(fig)

            # 6. 预测模块
            st.subheader("4. 在线预测")
            input_data = {}
            for col in feature_cols:
                input_data[col] = st.number_input(f"输入{col}的值", value=float(X[col].mean()))
            if st.button("预测"):
                input_df = pd.DataFrame([input_data])
                pred = model.predict(input_df)
                st.write(f"**预测结果**: {pred[0]}")