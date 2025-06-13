# main_app_streamlit.py

import streamlit as st
import pandas as pd
from automl_pipeline import DataPreprocessor, ModelTrainer, PipelineSaver

st.set_page_config(page_title="AutoML System", layout="wide")
st.title("🤖 AutoML Model Builder")

# Sidebar Inputs
task_type = st.sidebar.selectbox("Select Task Type", ["classification", "regression"])

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
target_col = st.sidebar.text_input("Enter Target Column Name")

model_options = ["LogisticRegression", "RandomForest", "XGBoost", "LinearRegression", "RandomForestRegressor", "XGBRegressor"]
model_select = st.sidebar.multiselect("Select Models (optional)", model_options)

run_button = st.sidebar.button("🚀 Run AutoML")

if uploaded_file and target_col and run_button:
    try:
        st.subheader("📄 Dataset Preview")
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        st.info("🔧 Preprocessing Data...")
        pre = DataPreprocessor(task_type=task_type, target_column=target_col)
        X, y = pre.preprocess(df)
        X_train, X_test, y_train, y_test = pre.scale_and_split(X, y)
        X_train, y_train = pre.balance_data(X_train, y_train)

        st.info("🧠 Training and Tuning Models...")
        trainer = ModelTrainer(task_type=task_type, models=model_select if model_select else 'all')
        trainer.select_models()
        trainer.tune_and_train(X_train, y_train, X_test, y_test)

        st.success("✅ Best Model Trained Successfully!")
        st.write(trainer.best_model)

        st.info("💾 Saving Pipeline...")
        saver = PipelineSaver()
        saver.save(trainer.best_model, pre)

        st.success("✅ Model, Encoders, and Scaler Saved!")
    
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

elif run_button:
    st.warning("⚠️ Please upload a dataset and provide the target column name.")
