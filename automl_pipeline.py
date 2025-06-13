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
    StandardScaler, RobustScaler, OneHotEncoder,
    OrdinalEncoder, LabelEncoder, PowerTransformer,
    QuantileTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso, ElasticNet
)
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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self, task_type, target_column):
        self.task_type = task_type
        self.target_column = target_column
        self.plots = []
    
    def analyze(self, df):
        results = {}
        df = df.copy()
        
        # Basic info
        results['shape'] = df.shape
        results['dtypes'] = df.dtypes.value_counts().to_dict()
        
        # Missing values
        missing = df.isnull().sum()
        results['missing_values'] = missing[missing > 0].sort_values(ascending=False).to_dict()
        
        # Identify feature types
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        elif self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        results['numeric_cols'] = numeric_cols
        results['categorical_cols'] = categorical_cols
        
        # Target analysis
        results['target_stats'] = self._analyze_target(df[self.target_column])
        
        # Numeric features analysis
        if numeric_cols:
            results['numeric_stats'] = self._analyze_numeric_features(df[numeric_cols])
        
        # Categorical features analysis
        if categorical_cols:
            results['categorical_stats'] = self._analyze_categorical_features(df[categorical_cols])
        
        # Correlation analysis for numeric features
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols + ([self.target_column] if self.target_column in df.select_dtypes(include=np.number).columns else [])].corr()
            results['correlation_matrix'] = corr_matrix.abs().mean().sort_values(ascending=False).to_dict()
            
            # Plot correlation heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlation Matrix')
            self.plots.append(fig)
        
        return results, self.plots
    
    def _analyze_target(self, target_series):
        stats = {}
        
        if pd.api.types.is_numeric_dtype(target_series):
            stats['type'] = 'numeric'
            stats.update({
                'min': target_series.min(),
                'max': target_series.max(),
                'mean': target_series.mean(),
                'median': target_series.median(),
                'std': target_series.std(),
                'skewness': target_series.skew(),
                'kurtosis': target_series.kurtosis()
            })
            
            # Distribution plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(target_series, kde=True, ax=ax)
            ax.set_title('Target Distribution (Numeric)')
            self.plots.append(fig)
            
            # Boxplot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=target_series, ax=ax)
            ax.set_title('Target Value Distribution')
            self.plots.append(fig)
        else:
            stats['type'] = 'categorical'
            value_counts = target_series.value_counts()
            stats.update({
                'unique_values': len(value_counts),
                'class_distribution': value_counts.to_dict(),
                'class_balance': 'Balanced' if max(value_counts)/min(value_counts) < 2 else 'Imbalanced'
            })
            
            # Bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            value_counts.plot(kind='bar', ax=ax)
            ax.set_title('Target Class Distribution')
            self.plots.append(fig)
            
            # Pie chart for <= 5 classes
            if len(value_counts) <= 5:
                fig, ax = plt.subplots(figsize=(8, 8))
                value_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                ax.set_title('Class Proportions')
                ax.set_ylabel('')
                self.plots.append(fig)
        
        return stats
    
    def _analyze_numeric_features(self, numeric_df):
        stats = {
            'descriptive': numeric_df.describe().to_dict(),
            'skewness': numeric_df.skew().to_dict(),
            'kurtosis': numeric_df.kurtosis().to_dict()
        }
        
        # Boxplots for all numeric features
        fig, ax = plt.subplots(figsize=(12, 8))
        numeric_df.boxplot(ax=ax)
        ax.set_title('Numeric Features Distribution')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        self.plots.append(fig)
        
        # Histograms for each numeric feature
        for col in numeric_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(numeric_df[col], kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
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
            
            # Plot for categorical features with <= 15 unique values
            if len(value_counts) <= 15:
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts.plot(kind='bar', ax=ax)
                ax.set_title(f'Distribution of {col}')
                self.plots.append(fig)
                
                # Pie chart for <= 5 unique values
                if len(value_counts) <= 5:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    value_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                    ax.set_title(f'Value Proportions for {col}')
                    ax.set_ylabel('')
                    self.plots.append(fig)
        
        return stats

class DataPreprocessor:
    def __init__(self, task_type, target_column, outlier_threshold=1.5, skew_threshold=1.0):
        self.task_type = task_type
        self.target_column = target_column
        self.outlier_threshold = outlier_threshold
        self.skew_threshold = skew_threshold
        self.scaler = None
        self.encoders = {}
        self.imputers = {}
        self.plots = []
    
    def preprocess(self, df):
        df = df.copy()
        
        # Initial cleaning
        df = self._clean_data(df)
        
        # Handle target
        y = self._process_target(df[self.target_column])
        X = df.drop(columns=[self.target_column])
        
        # Process features
        X = self._process_features(X)
        
        return X, y
    
    def _clean_data(self, df):
        # Drop completely empty rows/columns
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        
        # Drop duplicate rows
        df.drop_duplicates(inplace=True)
        
        # Drop columns with high percentage of missing values
        missing_percent = df.isnull().mean()
        cols_to_drop = missing_percent[missing_percent > 0.8].index.tolist()
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
        
        return df
    
    def _process_target(self, y):
        if self.task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.encoders['target'] = le
            return pd.Series(y_encoded, index=y.index)
        return y
    
    def _process_features(self, X):
        # Identify feature types
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Process numeric features
        if numeric_cols:
            X[numeric_cols] = self._process_numeric_features(X[numeric_cols])
        
        # Process categorical features
        if categorical_cols:
            X = self._process_categorical_features(X, categorical_cols)
        
        return X
    
    def _process_numeric_features(self, X_num):
        # Impute missing values
        num_imputer = SimpleImputer(strategy='median')
        X_num_imputed = pd.DataFrame(
            num_imputer.fit_transform(X_num),
            columns=X_num.columns,
            index=X_num.index
        )
        self.imputers['numeric'] = num_imputer
        
        # Handle outliers
        X_num_no_outliers, outlier_info = self._handle_outliers(X_num_imputed)
        
        # Transform skewed features
        X_num_transformed = self._transform_skewed_features(X_num_no_outliers)
        
        # Scale features
        self.scaler = StandardScaler()
        X_num_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_num_transformed),
            columns=X_num.columns,
            index=X_num.index
        )
        
        return X_num_scaled
    
    def _handle_outliers(self, X_num):
        outlier_info = {}
        X_clean = X_num.copy()
        
        for col in X_num.columns:
            q1 = X_num[col].quantile(0.25)
            q3 = X_num[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (self.outlier_threshold * iqr)
            upper_bound = q3 + (self.outlier_threshold * iqr)
            
            outliers = X_num[(X_num[col] < lower_bound) | (X_num[col] > upper_bound)]
            if not outliers.empty:
                # Cap outliers
                X_clean[col] = X_num[col].clip(lower_bound, upper_bound)
                outlier_info[col] = {
                    'num_outliers': len(outliers),
                    'method': 'capping'
                }
                
                # Create before/after plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                sns.boxplot(x=X_num[col], ax=ax1)
                ax1.set_title(f'Before: {col}')
                sns.boxplot(x=X_clean[col], ax=ax2)
                ax2.set_title(f'After: {col}')
                self.plots.append(fig)
        
        return X_clean, outlier_info
    
    def _transform_skewed_features(self, X_num):
        X_transformed = X_num.copy()
        
        for col in X_num.columns:
            skewness = X_num[col].skew()
            
            if abs(skewness) > self.skew_threshold:
                # Positive skew
                if skewness > 0:
                    # Try both transformations
                    bc_transformed, _ = stats.boxcox(X_num[col] + 1)  # Add 1 to handle zeros
                    qt_transformed = QuantileTransformer(output_distribution='normal').fit_transform(X_num[[col]])
                    
                    # Choose the one with lower absolute skewness
                    bc_skew = stats.skew(bc_transformed)
                    qt_skew = stats.skew(qt_transformed.flatten())
                    
                    if abs(bc_skew) < abs(qt_skew):
                        X_transformed[col] = bc_transformed
                        best_method = 'boxcox'
                    else:
                        X_transformed[col] = qt_transformed.flatten()
                        best_method = 'quantile'
                
                # Negative skew
                else:
                    # Try both transformations
                    yj_transformed = PowerTransformer(method='yeo-johnson').fit_transform(X_num[[col]])
                    qt_transformed = QuantileTransformer(output_distribution='normal').fit_transform(X_num[[col]])
                    
                    # Choose the one with lower absolute skewness
                    yj_skew = stats.skew(yj_transformed.flatten())
                    qt_skew = stats.skew(qt_transformed.flatten())
                    
                    if abs(yj_skew) < abs(qt_skew):
                        X_transformed[col] = yj_transformed.flatten()
                        best_method = 'yeo-johnson'
                    else:
                        X_transformed[col] = qt_transformed.flatten()
                        best_method = 'quantile'
                
                # Create transformation plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                sns.histplot(X_num[col], kde=True, ax=ax1)
                ax1.set_title(f'Original: {col} (skew={skewness:.2f})')
                sns.histplot(X_transformed[col], kde=True, ax=ax2)
                ax2.set_title(f'Transformed: {best_method} (skew={stats.skew(X_transformed[col]):.2f})')
                self.plots.append(fig)
        
        return X_transformed
    
    def _process_categorical_features(self, X, categorical_cols):
        # Identify low/high cardinality features
        low_card_cols = [col for col in categorical_cols if X[col].nunique() < 10]
        high_card_cols = [col for col in categorical_cols if X[col].nunique() >= 10]
        
        # Process low cardinality with one-hot encoding
        if low_card_cols:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            ohe_features = pd.DataFrame(
                ohe.fit_transform(X[low_card_cols]),
                columns=ohe.get_feature_names_out(low_card_cols),
                index=X.index
            )
            self.encoders['onehot'] = ohe
            
            # Process high cardinality with ordinal encoding
            if high_card_cols:
                ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                ordinal_features = pd.DataFrame(
                    ordinal.fit_transform(X[high_card_cols]),
                    columns=high_card_cols,
                    index=X.index
                )
                self.encoders['ordinal'] = ordinal
                
                # Combine features
                X_processed = pd.concat([
                    X.drop(columns=categorical_cols),
                    ohe_features,
                    ordinal_features
                ], axis=1)
            else:
                X_processed = pd.concat([
                    X.drop(columns=categorical_cols),
                    ohe_features
                ], axis=1)
        else:
            # Only high cardinality
            ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            ordinal_features = pd.DataFrame(
                ordinal.fit_transform(X[high_card_cols]),
                columns=high_card_cols,
                index=X.index
            )
            self.encoders['ordinal'] = ordinal
            X_processed = pd.concat([
                X.drop(columns=categorical_cols),
                ordinal_features
            ], axis=1)
        
        return X_processed
    
    def split_data(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if self.task_type == "classification" else None
        )
        return X_train, X_test, y_train, y_test
    
    def balance_data(self, X_train, y_train):
        if self.task_type == "classification":
            class_counts = np.bincount(y_train)
            if min(class_counts) / max(class_counts) < 0.5:
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(X_train, y_train)
                self.encoders['smote'] = smote
                return X_res, y_res
        return X_train, y_train

class FeatureEngineer:
    def __init__(self, task_type):
        self.task_type = task_type
        self.selector = None
        self.pca = None
        self.feature_importance_plot = None
    
    def transform(self, X):
        # Feature selection
        if X.shape[1] > 20:
            X = self._select_features(X)
        
        # Dimensionality reduction
        if X.shape[1] > 50:
            X = self._reduce_dimensions(X)
        
        return X
    
    def _select_features(self, X):
        k = min(20, X.shape[1] // 2)
        if self.task_type == "classification":
            selector = SelectKBest(f_classif, k=k)
        else:
            selector = SelectKBest(f_regression, k=k)
        
        X_selected = selector.fit_transform(X, y=None)  # y not needed for unsupervised selection
        self.selector = selector
        
        # Create feature importance plot
        if hasattr(selector, 'scores_'):
            fig, ax = plt.subplots(figsize=(12, 8))
            features = X.columns
            scores = selector.scores_
            sorted_idx = np.argsort(scores)[-k:]
            ax.barh(range(len(sorted_idx)), scores[sorted_idx])
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([features[i] for i in sorted_idx])
            ax.set_title('Feature Importance Scores')
            self.feature_importance_plot = fig
        
        return pd.DataFrame(X_selected, index=X.index)
    
    def _reduce_dimensions(self, X):
        n_components = min(50, X.shape[1] // 2)
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        return pd.DataFrame(X_pca, index=X.index)

class ModelTrainer:
    def __init__(self, task_type, models, metrics):
        self.task_type = task_type
        self.models_to_train = models
        self.metrics = metrics
        self.models = []
        self.best_model = None
        self.best_score = -np.inf if task_type == "classification" else np.inf
        self.best_model_name = ""
        self.results = {}
        self.training_logs = []
    
    def train(self, X_train, y_train, X_test, y_test):
        self._initialize_models()
        
        for model_name, model, params in self.models:
            self.training_logs.append(f"⚙️ Training {model_name}...")
            
            try:
                if X_train.shape[0] <= 100000:
                    search = GridSearchCV(model, params, cv=5, n_jobs=-1, 
                                        scoring=self._get_scoring_metric())
                else:
                    search = RandomizedSearchCV(model, params, n_iter=10, cv=5, 
                                             n_jobs=-1, random_state=42,
                                             scoring=self._get_scoring_metric())
                
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                
                # Evaluate
                scores = self._evaluate_model(best_model, X_test, y_test)
                self.results[model_name] = scores
                
                # Update best model
                main_metric = self.metrics[0]
                if self._is_better(scores[main_metric], self.best_score):
                    self.best_score = scores[main_metric]
                    self.best_model = best_model
                    self.best_model_name = model_name
                
                self.training_logs.append(
                    f"✅ {model_name} - {main_metric}: {scores[main_metric]:.4f} "
                    f"(Best params: {search.best_params_})"
                )
            
            except Exception as e:
                self.training_logs.append(f"❌ Failed to train {model_name}: {str(e)}")
    
    def _initialize_models(self):
        classification_models = {
            'LogisticRegression': (
                LogisticRegression(max_iter=1000),
                {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2']}
            ),
            'DecisionTreeClassifier': (
                DecisionTreeClassifier(),
                {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
            ),
            'RandomForestClassifier': (
                RandomForestClassifier(),
                {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, None]}
            ),
            'XGBClassifier': (
                XGBClassifier(eval_metric='logloss', use_label_encoder=False),
                {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
            ),
            'GradientBoostingClassifier': (
                GradientBoostingClassifier(),
                {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
            ),
            'SVC': (
                SVC(probability=True),
                {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            ),
            'KNeighborsClassifier': (
                KNeighborsClassifier(),
                {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
            )
        }
        
        regression_models = {
            'LinearRegression': (
                LinearRegression(),
                {'fit_intercept': [True, False]}
            ),
            'DecisionTreeRegressor': (
                DecisionTreeRegressor(),
                {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
            ),
            'RandomForestRegressor': (
                RandomForestRegressor(),
                {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, None]}
            ),
            'XGBRegressor': (
                XGBRegressor(),
                {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
            ),
            'GradientBoostingRegressor': (
                GradientBoostingRegressor(),
                {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
            ),
            'SVR': (
                SVR(),
                {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            ),
            'KNeighborsRegressor': (
                KNeighborsRegressor(),
                {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
            )
        }
        
        all_models = classification_models if self.task_type == "classification" else regression_models
        
        if 'all' in self.models_to_train:
            self.models = [(name, model, params) for name, (model, params) in all_models.items()]
        else:
            self.models = [
                (name, all_models[name][0], all_models[name][1]) 
                for name in self.models_to_train 
                if name in all_models
            ]
    
    def _get_scoring_metric(self):
        if self.task_type == "classification":
            return 'accuracy' if 'accuracy' in self.metrics else self.metrics[0]
        else:
            return 'neg_mean_squared_error' if 'mse' in self.metrics else f"neg_{self.metrics[0]}"
    
    def _evaluate_model(self, model, X_test, y_test):
        scores = {}
        y_pred = model.predict(X_test)
        
        if self.task_type == "classification":
            for metric in self.metrics:
                if metric == 'accuracy':
                    scores[metric] = accuracy_score(y_test, y_pred)
                elif metric == 'f1':
                    scores[metric] = f1_score(y_test, y_pred)
                elif metric == 'roc_auc':
                    if len(np.unique(y_test)) == 2:  # Binary classification
                        y_prob = model.predict_proba(X_test)[:, 1]
                        scores[metric] = roc_auc_score(y_test, y_prob)
                    else:
                        scores[metric] = np.nan
                elif metric == 'precision':
                    scores[metric] = precision_score(y_test, y_pred)
                elif metric == 'recall':
                    scores[metric] = recall_score(y_test, y_pred)
        else:
            for metric in self.metrics:
                if metric == 'mse':
                    scores[metric] = mean_squared_error(y_test, y_pred)
                elif metric == 'mae':
                    scores[metric] = mean_absolute_error(y_test, y_pred)
                elif metric == 'r2':
                    scores[metric] = r2_score(y_test, y_pred)
                elif metric == 'rmse':
                    scores[metric] = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return scores
    
    def _is_better(self, new_score, current_best):
        if self.task_type == "classification":
            return new_score > current_best
        else:
            return new_score < current_best
    
    def plot_model_comparison(self):
        if not self.results:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if self.task_type == "classification":
            metric = 'accuracy' if 'accuracy' in self.metrics else self.metrics[0]
            scores = [self.results[m][metric] for m in self.results]
            ax.barh(list(self.results.keys()), scores)
            ax.set_xlabel(metric)
            ax.set_title('Model Comparison (Higher is better)')
        else:
            metric = 'mse' if 'mse' in self.metrics else self.metrics[0]
            scores = [self.results[m][metric] for m in self.results]
            ax.barh(list(self.results.keys()), scores)
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
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(indices)), importances[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([f"Feature {i}" for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Feature Importance ({self.best_model_name})')
        
        return fig

class PipelineSaver:
    def __init__(self, path='automl_pipeline'):
        self.path = path
        os.makedirs(self.path, exist_ok=True)
    
    def save(self, best_model, preprocessor, feature_engineer, all_models=None):
        # Save best model
        joblib.dump(best_model, os.path.join(self.path, 'best_model.pkl'))
        
        # Save preprocessor artifacts
        if preprocessor.scaler:
            joblib.dump(preprocessor.scaler, os.path.join(self.path, 'scaler.pkl'))
        
        for name, encoder in preprocessor.encoders.items():
            joblib.dump(encoder, os.path.join(self.path, f'{name}_encoder.pkl'))
        
        if hasattr(preprocessor, 'imputers'):
            for name, imputer in preprocessor.imputers.items():
                joblib.dump(imputer, os.path.join(self.path, f'{name}_imputer.pkl'))
        
        # Save feature engineering artifacts
        if feature_engineer.selector:
            joblib.dump(feature_engineer.selector, os.path.join(self.path, 'feature_selector.pkl'))
        
        if feature_engineer.pca:
            joblib.dump(feature_engineer.pca, os.path.join(self.path, 'pca.pkl'))
        
        # Save all models if provided
        if all_models:
            for model_name, model in all_models.items():
                joblib.dump(model, os.path.join(self.path, f'{model_name}_model.pkl'))
        
        # Create zip file
        zip_path = os.path.join(self.path, 'automl_pipeline.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self.path):
                for file in files:
                    if file.endswith('.pkl'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.path)
                        zipf.write(file_path, arcname)
        
        return zip_path
