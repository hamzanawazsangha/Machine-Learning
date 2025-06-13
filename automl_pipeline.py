import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import zipfile
import os
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, 
    OrdinalEncoder, LabelEncoder, PowerTransformer,
    QuantileTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, confusion_matrix,
    precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self, task_type, target_column):
        self.task_type = task_type
        self.target_column = target_column
        self.plots = []
    
    def analyze(self, df):
        results = {}
        
        # Basic dataset info
        results['shape'] = df.shape
        results['dtypes'] = df.dtypes.value_counts().to_dict()
        
        # Identify feature types
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        elif self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        results['numeric_cols'] = numeric_cols
        results['categorical_cols'] = categorical_cols
        
        # Missing values analysis
        missing = df.isnull().sum()
        results['missing_values'] = missing[missing > 0].sort_values(ascending=False).to_dict()
        
        # Target analysis
        results['target_stats'] = self._analyze_target(df[self.target_column])
        
        # Numeric features analysis
        if numeric_cols:
            results['numeric_stats'] = self._analyze_numeric_features(df[numeric_cols])
        
        # Categorical features analysis
        if categorical_cols:
            results['categorical_stats'] = self._analyze_categorical_features(df[categorical_cols])
        
        return results, self.plots
    
    def _analyze_target(self, target_series):
        stats = {}
        
        if pd.api.types.is_numeric_dtype(target_series):
            # For numeric targets
            stats['type'] = 'numeric'
            stats.update({
                'min': target_series.min(),
                'max': target_series.max(),
                'mean': target_series.mean(),
                'median': target_series.median(),
                'std': target_series.std(),
                'skewness': target_series.skew()
            })
            
            # Create distribution plot
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(target_series, kde=True, ax=ax)
            ax.set_title('Target Distribution (Numeric)')
            self.plots.append(fig)
            
            # Create boxplot
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x=target_series, ax=ax)
            ax.set_title('Target Boxplot')
            self.plots.append(fig)
        else:
            # For categorical targets
            stats['type'] = 'categorical'
            value_counts = target_series.value_counts()
            stats.update({
                'unique_values': len(value_counts),
                'class_distribution': value_counts.to_dict()
            })
            
            # Create pie chart for small number of categories
            if len(value_counts) <= 10:
                fig, ax = plt.subplots(figsize=(8, 5))
                value_counts.plot.pie(autopct='%1.1f%%', ax=ax)
                ax.set_title('Target Distribution (Categorical)')
                ax.set_ylabel('')
                self.plots.append(fig)
            
            # Always create bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            value_counts.plot.bar(ax=ax)
            ax.set_title('Target Class Distribution')
            self.plots.append(fig)
        
        return stats
    
    def _analyze_numeric_features(self, numeric_df):
        stats = {
            'descriptive_stats': numeric_df.describe().to_dict(),
            'skewness': numeric_df.skew().to_dict(),
            'kurtosis': numeric_df.kurtosis().to_dict()
        }
        
        # Create boxplots for numeric features
        fig, ax = plt.subplots(figsize=(10, 6))
        numeric_df.boxplot(ax=ax)
        ax.set_title('Numeric Features Distribution')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        self.plots.append(fig)
        
        # Create correlation heatmap if multiple numeric features
        if len(numeric_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', 
                        center=0, ax=ax)
            ax.set_title('Numeric Features Correlation')
            self.plots.append(fig)
        
        return stats
    
    def _analyze_categorical_features(self, categorical_df):
        stats = {}
        
        for col in categorical_df.columns:
            value_counts = categorical_df[col].value_counts()
            stats[col] = {
                'unique_values': len(value_counts),
                'top_value': value_counts.index[0],
                'top_frequency': value_counts.iloc[0],
                'null_count': categorical_df[col].isnull().sum()
            }
            
            # Create plots for categorical features with reasonable cardinality
            if len(value_counts) <= 15:
                # Bar plot
                fig, ax = plt.subplots(figsize=(8, 5))
                value_counts.plot.bar(ax=ax)
                ax.set_title(f'Distribution of {col}')
                self.plots.append(fig)
                
                # Pie chart for features with <= 5 categories
                if len(value_counts) <= 5:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    value_counts.plot.pie(autopct='%1.1f%%', ax=ax)
                    ax.set_title(f'{col} Value Proportions')
                    ax.set_ylabel('')
                    self.plots.append(fig)
        
        return stats

class DataPreprocessor:
    def __init__(self, task_type, target_column):
        self.task_type = task_type
        self.target_column = target_column
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.encoders = {}
        self.feature_names = []
    
    def preprocess(self, df):
        df = df.copy()
        
        # Handle missing values in target
        df = df.dropna(subset=[self.target_column])
        
        # Separate features and target
        y = self._process_target(df[self.target_column])
        X = df.drop(columns=[self.target_column])
        
        # Identify feature types
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', self._get_encoder(categorical_cols, X))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Fit and transform the data
        X_processed = preprocessor.fit_transform(X)
        
        # Store the transformers for later use
        self.numeric_imputer = preprocessor.named_transformers_['num'].named_steps['imputer']
        self.scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
        self.categorical_imputer = preprocessor.named_transformers_['cat'].named_steps['imputer']
        
        # Get feature names after transformation
        numeric_features = numeric_cols
        if hasattr(preprocessor.named_transformers_['cat'].named_steps['encoder'], 'get_feature_names_out'):
            categorical_features = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols)
        else:
            categorical_features = categorical_cols
        
        self.feature_names = numeric_features + list(categorical_features)
        
        return X_processed, y
    
    def _process_target(self, target_series):
        if self.task_type == "classification" and not pd.api.types.is_numeric_dtype(target_series):
            self.encoders['target'] = LabelEncoder()
            return self.encoders['target'].fit_transform(target_series)
        return target_series.values
    
    def _get_encoder(self, categorical_cols, X):
        # Use OneHotEncoder for low cardinality, Ordinal for high cardinality
        low_card_cols = [col for col in categorical_cols if X[col].nunique() < 10]
        high_card_cols = [col for col in categorical_cols if X[col].nunique() >= 10]
        
        if low_card_cols and high_card_cols:
            # Mixed case - use ColumnTransformer
            return ColumnTransformer([
                ('onehot', OneHotEncoder(handle_unknown='ignore'), low_card_cols),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), high_card_cols)
            ])
        elif low_card_cols:
            # All low cardinality - use OneHot
            return OneHotEncoder(handle_unknown='ignore')
        else:
            # All high cardinality - use Ordinal
            return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    def split_data(self, X, y, test_size=0.2):
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42,
            stratify=y if self.task_type == "classification" else None
        )

class ModelTrainer:
    def __init__(self, task_type, models, metrics):
        self.task_type = task_type
        self.models = models
        self.metrics = metrics
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf if task_type == "classification" else np.inf
        self.best_model_name = ""
    
    def train(self, X_train, y_train, X_test, y_test):
        model_configs = self._get_model_configs()
        
        for name in self.models:
            if name in model_configs:
                model, params = model_configs[name]
                
                # Select search strategy based on data size
                if X_train.shape[0] < 10000:
                    search = GridSearchCV(model, params, cv=5, 
                                         scoring=self._get_scoring(),
                                         n_jobs=-1)
                else:
                    search = RandomizedSearchCV(model, params, n_iter=10, cv=3,
                                              scoring=self._get_scoring(),
                                              n_jobs=-1, random_state=42)
                
                search.fit(X_train, y_train)
                
                # Evaluate on test set
                scores = self._evaluate(search.best_estimator_, X_test, y_test)
                self.results[name] = {
                    'model': search.best_estimator_,
                    'params': search.best_params_,
                    'scores': scores
                }
                
                # Update best model
                main_metric = self.metrics[0]
                current_score = scores[main_metric]
                if self._is_better(current_score, self.best_score):
                    self.best_score = current_score
                    self.best_model = search.best_estimator_
                    self.best_model_name = name
    
    def _get_model_configs(self):
        common_regression_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, None],
            'min_samples_split': [2, 5, 10]
        }
        
        if self.task_type == "classification":
            return {
                'LogisticRegression': (
                    LogisticRegression(max_iter=1000),
                    {'C': [0.1, 1, 10], 'penalty': ['l2']}
                ),
                'RandomForestClassifier': (
                    RandomForestClassifier(),
                    common_regression_params
                ),
                'XGBClassifier': (
                    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                    {'learning_rate': [0.01, 0.1], **common_regression_params}
                ),
                'SVC': (
                    SVC(probability=True),
                    {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
                )
            }
        else:
            return {
                'LinearRegression': (
                    LinearRegression(),
                    {'fit_intercept': [True, False]}
                ),
                'RandomForestRegressor': (
                    RandomForestRegressor(),
                    common_regression_params
                ),
                'XGBRegressor': (
                    XGBRegressor(),
                    {'learning_rate': [0.01, 0.1], **common_regression_params}
                ),
                'SVR': (
                    SVR(),
                    {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
                )
            }
    
    def _get_scoring(self):
        if self.task_type == "classification":
            return 'accuracy' if 'accuracy' in self.metrics else self.metrics[0]
        return 'neg_mean_squared_error' if 'mse' in self.metrics else f"neg_{self.metrics[0]}"
    
    def _evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        scores = {}
        
        for metric in self.metrics:
            if metric == 'accuracy':
                scores[metric] = accuracy_score(y_test, y_pred)
            elif metric == 'f1':
                scores[metric] = f1_score(y_test, y_pred)
            elif metric == 'roc_auc':
                if hasattr(model, 'predict_proba'):
                    scores[metric] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            elif metric == 'precision':
                scores[metric] = precision_score(y_test, y_pred)
            elif metric == 'recall':
                scores[metric] = recall_score(y_test, y_pred)
            elif metric == 'mse':
                scores[metric] = mean_squared_error(y_test, y_pred)
            elif metric == 'mae':
                scores[metric] = mean_absolute_error(y_test, y_pred)
            elif metric == 'r2':
                scores[metric] = r2_score(y_test, y_pred)
        
        return scores
    
    def _is_better(self, new_score, current_best):
        if self.task_type == "classification":
            return new_score > current_best
        return new_score < current_best
    
    def plot_results(self):
        if not self.results:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        metric = self.metrics[0]
        model_names = list(self.results.keys())
        scores = [self.results[name]['scores'][metric] for name in model_names]
        
        if self.task_type == "classification":
            ax.barh(model_names, scores)
            ax.set_xlabel(metric)
            ax.set_title('Model Comparison (Higher is better)')
        else:
            ax.barh(model_names, scores)
            ax.set_xlabel(metric)
            ax.set_title('Model Comparison (Lower is better)')
        
        return fig
    
    def plot_confusion_matrix(self, X_test, y_test):
        if self.task_type != "classification" or self.best_model is None:
            return None
            
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix ({self.best_model_name})')
        
        return fig
    
    def plot_feature_importance(self):
        if self.best_model is None or not hasattr(self.best_model, 'feature_importances_'):
            return None
            
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20 features
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Feature Importance ({self.best_model_name})')
        
        return fig

class PipelineSaver:
    def __init__(self):
        self.artifacts = {}
    
    def add_artifact(self, name, obj):
        self.artifacts[name] = obj
    
    def save(self, output_dir='pipeline_artifacts'):
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each artifact
        saved_paths = []
        for name, obj in self.artifacts.items():
            path = os.path.join(output_dir, f"{name}.pkl")
            joblib.dump(obj, path)
            saved_paths.append(path)
        
        # Create zip file
        zip_path = os.path.join(output_dir, 'pipeline.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in saved_paths:
                zipf.write(file_path, os.path.basename(file_path))
        
        return zip_path
