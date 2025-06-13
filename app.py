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
    metric_options = ["accuracy", "roc_auc", "confusion_matrix", "classification_report"]
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
        trainer = ModelTrainer(task_type=task_type, models=model_select if model_select else 'all', eval_metrics=metrics)
        trainer.select_models()

        progress_bar = st.progress(0)
        total_models = len(trainer.models)
        trained_models_info = []

        for i, model in enumerate(trainer.models):
            st.write(f"🔄 Tuning Model: `{model}`")
            trainer.tune_model(model, X_train, y_train, X_test, y_test)
            scores = trainer.evaluate_model(model, X_test, y_test)
            trained_models_info.append((model, scores))
            progress_bar.progress((i + 1) / total_models)

        st.success("✅ All Models Trained!")

        st.subheader("📊 Evaluation Results")
        for model, score in trained_models_info:
            st.write(f"Model: {model}")
            st.json(score)

        st.success(f"🏆 Best Model: {trainer.best_model_name}")
        st.write(trainer.best_model)

        st.info("💾 Saving All Models and Pipeline...")
        saver = PipelineSaver()
        all_saved_paths = saver.save_all(trainer.models_dict, pre)

        zip_path = "automl_models_bundle.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in all_saved_paths:
                zipf.write(file_path, arcname=os.path.basename(file_path))

        st.download_button("⬇️ Download All Models and Artifacts", open(zip_path, "rb"), file_name=zip_path)

        st.subheader("🧪 How to Use the Saved Pipeline")
        st.code('''
import pickle
import pandas as pd

# Load files
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Preprocess new data (same steps as training)
new_data = pd.read_csv("new_data.csv")
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
        ''', language="python")

        # Clean up temporary files
        for path in all_saved_paths:
            os.remove(path)
        os.remove(zip_path)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

elif run_button:
    st.warning("⚠️ Please upload a dataset and provide the target column name.")
