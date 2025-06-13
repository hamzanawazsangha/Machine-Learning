# main_app_streamlit.py

import streamlit as st
import pandas as pd
import zipfile
import shutil
import os
from automl_pipeline import DataPreprocessor, ModelTrainer, PipelineSaver

st.set_page_config(page_title="AutoML System", layout="wide")
st.title("🤖 AutoML Model Builder")

# Sidebar Inputs
task_type = st.sidebar.selectbox("Select Task Type", ["classification", "regression"])

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
target_col = st.sidebar.text_input("Enter Target Column Name")

# Dynamically choose models based on task type
if task_type == "classification":
    model_options = [
        "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier",
        "XGBClassifier", "GradientBoostingClassifier", "SVC", "KNeighborsClassifier"
    ]
    metric_options = ["accuracy", "roc_auc", "f1_score", "confusion_matrix"]
elif task_type == "regression":
    model_options = [
        "LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor",
        "XGBRegressor", "GradientBoostingRegressor", "SVR", "KNeighborsRegressor",
        "Lasso", "Ridge", "ElasticNet"
    ]
    metric_options = ["mse", "mae", "r2", "rmse"]
else:
    model_options = []
    metric_options = []

model_select = st.sidebar.multiselect("Select Models (optional)", model_options)
eval_metrics = st.sidebar.multiselect("Select Evaluation Metrics", metric_options)

run_button = st.sidebar.button("🚀 Run AutoML")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.write(df.head())

if uploaded_file and target_col and run_button:
    try:
        st.info("🔧 Preprocessing Data...")
        pre = DataPreprocessor(task_type=task_type, target_column=target_col)
        X, y = pre.preprocess(df)
        st.success("✅ Data Preprocessed Successfully!")

        st.info("🧬 Performing Feature Engineering and Scaling...")
        X_train, X_test, y_train, y_test = pre.scale_and_split(X, y)
        X_train, y_train = pre.balance_data(X_train, y_train)
        st.success("✅ Feature Engineering Done!")

        st.info("🧠 Training and Tuning Models...")
        trainer = ModelTrainer(task_type=task_type, models=model_select if model_select else 'all', metrics=eval_metrics)
        trainer.select_models()

        progress_bar = st.progress(0)
        total_models = len(trainer.models)
        
        # Train and tune models
        trainer.tune_and_train(X_train, y_train, X_test, y_test)

        st.success("✅ All Models Trained!")

        st.subheader("📊 Evaluation Results")
        for model_name, score in trainer.model_scores.items():
            st.write(f"Model: {model_name}")
            st.write(f"Score: {score:.4f}")

        if trainer.best_model is not None:
            st.success(f"🏆 Best Model: {type(trainer.best_model).__name__} with score {trainer.best_score:.4f}")
        else:
            st.warning("⚠️ No model was successfully trained.")

        st.info("💾 Saving All Models and Pipeline...")
        saver = PipelineSaver()
        zip_path = saver.save(trainer.best_model, pre)

        with open(zip_path, "rb") as f:
            st.download_button(
                "⬇️ Download All Models and Artifacts",
                f,
                file_name="automl_artifacts.zip",
                mime="application/zip"
            )

        st.subheader("🧪 How to Use the Saved Pipeline")
        st.code('''
import joblib
import pandas as pd

# Load files
scaler = joblib.load("scaler.pkl")
model = joblib.load("best_model.pkl")

# Preprocess new data (same steps as training)
new_data = pd.read_csv("new_data.csv")
# Apply the same preprocessing steps you used on your training data
# Then scale:
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
        ''', language="python")

        # Clean up temporary files
        os.remove(zip_path)
        shutil.rmtree(saver.path)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        st.error(str(e))

elif run_button:
    st.warning("⚠️ Please upload a dataset and provide the target column name.")
