import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
import joblib
from automl_pipeline import DataAnalyzer, DataPreprocessor, FeatureEngineer, ModelTrainer, PipelineSaver

st.set_page_config(page_title="Enhanced AutoML System", layout="wide")
st.title("🤖 Advanced AutoML Pipeline")

# Sidebar Inputs
with st.sidebar:
    st.header("Configuration")
    task_type = st.selectbox("Select Task Type", ["classification", "regression"])
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_col = st.selectbox("Select Target Column", df.columns)
        
        # Model selection
        st.subheader("Model Selection")
        if task_type == "classification":
            model_options = [
                "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier",
                "XGBClassifier", "GradientBoostingClassifier", "SVC", "KNeighborsClassifier"
            ]
            metric_options = ["accuracy", "f1", "roc_auc", "precision", "recall"]
        else:
            model_options = [
                "LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor",
                "XGBRegressor", "GradientBoostingRegressor", "SVR", "KNeighborsRegressor"
            ]
            metric_options = ["mse", "mae", "r2", "rmse"]
        
        model_select = st.multiselect("Select Models", model_options, default=model_options[:3])
        eval_metrics = st.multiselect("Select Evaluation Metrics", metric_options, default=metric_options[0])
        
        # Advanced options
        st.subheader("Advanced Options")
        outlier_threshold = st.slider("Outlier Threshold (IQR multiplier)", 1.5, 5.0, 1.5, 0.1)
        skew_threshold = st.slider("Skewness Threshold for Transformation", 0.1, 2.0, 0.5, 0.1)
        test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)
        
        run_button = st.button("🚀 Run AutoML Pipeline")

# Main Content
if uploaded_file:
    st.subheader("📊 Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())

if uploaded_file and target_col and run_button:
    try:
        # Initialize components
        analyzer = DataAnalyzer(task_type=task_type, target_column=target_col)
        preprocessor = DataPreprocessor(task_type=task_type, target_column=target_col,
                                      outlier_threshold=outlier_threshold,
                                      skew_threshold=skew_threshold)
        feature_engineer = FeatureEngineer(task_type=task_type)
        trainer = ModelTrainer(task_type=task_type, models=model_select, metrics=eval_metrics)
        saver = PipelineSaver()

        # Tab layout
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Data Analysis", "Preprocessing", "Feature Engineering", 
            "Model Training", "Results"
        ])

        with tab1:
            st.info("🔍 Analyzing Data...")
            # Fix: Properly unpack the return values
            analysis_results, analysis_plots = analyzer.analyze(df)
            
            # Show analysis results
            st.subheader("Data Summary")
            st.write(analysis_results['summary'])
            
            # Visualizations
            st.subheader("Data Visualizations")
            for plot in analysis_plots:
                st.pyplot(plot)
                
            # Additional visualizations
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Missing Values")
                fig, ax = plt.subplots()
                sns.heatmap(df.isnull(), cbar=False, ax=ax)
                st.pyplot(fig)
                
            with col2:
                st.subheader("Target Distribution")
                fig, ax = plt.subplots()
                if task_type == "classification":
                    df[target_col].value_counts().plot(kind='bar', ax=ax)
                else:
                    sns.histplot(df[target_col], kde=True, ax=ax)
                st.pyplot(fig)
            
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) > 1:
                st.subheader("Correlation Matrix")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", ax=ax)
                st.pyplot(fig)

        with tab2:
            st.info("🔧 Preprocessing Data...")
            X, y = preprocessor.preprocess(df)
            
            # Show preprocessing results
            st.subheader("Preprocessing Summary")
            st.write(f"Original shape: {df.shape}")
            st.write(f"Processed shape: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Outlier visualization
            if hasattr(preprocessor, 'outlier_plots') and preprocessor.outlier_plots:
                st.subheader("Outlier Treatment")
                cols = st.columns(2)
                for i, plot in enumerate(preprocessor.outlier_plots):
                    with cols[i % 2]:
                        st.pyplot(plot)
            
            # Transformation visualization
            if hasattr(preprocessor, 'transformation_plots') and preprocessor.transformation_plots:
                st.subheader("Feature Transformation")
                cols = st.columns(2)
                for i, plot in enumerate(preprocessor.transformation_plots):
                    with cols[i % 2]:
                        st.pyplot(plot)

        with tab3:
            st.info("🛠️ Feature Engineering...")
            X = feature_engineer.transform(X)
            
            st.subheader("Engineered Features")
            st.dataframe(X.head())
            
            if hasattr(feature_engineer, 'feature_importance_plot'):
                st.subheader("Feature Importance")
                st.pyplot(feature_engineer.feature_importance_plot)

        with tab4:
            st.info("🧠 Training Models...")
            X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=test_size)
            
            # Balance data if classification
            if task_type == "classification":
                X_train, y_train = preprocessor.balance_data(X_train, y_train)
            
            trainer.train(X_train, y_train, X_test, y_test)
            
            st.subheader("Training Progress")
            for log in trainer.training_logs:
                st.write(log)

        with tab5:
            st.info("📊 Evaluating Results...")
            
            # Model comparison
            st.subheader("Model Comparison")
            fig = trainer.plot_model_comparison()
            if fig:  # Check if figure exists before trying to display
                st.pyplot(fig)
            
            # Best model details
            st.subheader("Best Model Performance")
            st.write(f"Best Model: {trainer.best_model_name}")
            st.write(f"Best Score: {trainer.best_score:.4f}")
            
            # Confusion matrix for classification
            if task_type == "classification":
                st.subheader("Confusion Matrix")
                cm_fig = trainer.plot_confusion_matrix(X_test, y_test)
                if cm_fig:
                    st.pyplot(cm_fig)
            
            # Feature importance
            if hasattr(trainer.best_model, 'feature_importances_'):
                st.subheader("Feature Importance")
                fi_fig = trainer.plot_feature_importance()
                if fi_fig:
                    st.pyplot(fi_fig)
            
            # Download artifacts
            st.subheader("Download Pipeline")
            try:
                zip_path = saver.save(
                    trainer.best_model, 
                    preprocessor, 
                    feature_engineer,
                    trainer.models
                )
                
                with open(zip_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Full Pipeline",
                        f,
                        file_name="automl_pipeline.zip",
                        mime="application/zip"
                    )
            except Exception as e:
                st.error(f"Failed to save pipeline: {str(e)}")

    except Exception as e:
        st.error(f"❌ Pipeline Error: {str(e)}")
        st.exception(e)
