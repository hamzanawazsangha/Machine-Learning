import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st
import zipfile
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from imblearn.over_sampling import SMOTE


class DataPreprocessor:
    def __init__(self, task_type, target_column):
        self.task_type = task_type
        self.target_column = target_column
        self.encoders = {}
        self.scaler = None
        self.smote = None

    def preprocess(self, df):
        df = df.copy()
        df.dropna(axis=0, how='all', inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.dropna(subset=[self.target_column], inplace=True)

        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])

        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        cat_imputer = SimpleImputer(strategy='most_frequent')
        num_imputer = SimpleImputer(strategy='mean')
        X[cat_cols] = pd.DataFrame(cat_imputer.fit_transform(X[cat_cols]), columns=cat_cols, index=X.index)
        X[num_cols] = pd.DataFrame(num_imputer.fit_transform(X[num_cols]), columns=num_cols, index=X.index)

        if self.task_type != 'unsupervised' and y.dtype == 'object':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)
            self.encoders['target'] = le

        if cat_cols:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            X_encoded = pd.DataFrame(ohe.fit_transform(X[cat_cols]), 
                                columns=ohe.get_feature_names_out(cat_cols), 
                                index=X.index)
            self.encoders['ohe'] = ohe
            X_final = pd.concat([X[num_cols], X_encoded], axis=1).loc[y.index]
        else:
            X_final = X[num_cols].loc[y.index]

        return X_final, y

    def scale_and_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def balance_data(self, X_train, y_train):
        if self.task_type == 'classification':
            if len(np.unique(y_train)) == 2:
                counts = np.bincount(y_train)
                if min(counts) / max(counts) < 0.6:
                    self.smote = SMOTE()
                    X_train, y_train = self.smote.fit_resample(X_train, y_train)
        return X_train, y_train


class ModelTrainer:
    def __init__(self, task_type='classification', models='all', metrics=None):
        self.task_type = task_type
        self.model_names = models
        self.models = []
        self.model_scores = {}
        self.best_model = None
        self.best_score = -float('inf') if task_type == 'classification' else float('inf')
        self.selected_metrics = metrics or []  # This stores the metrics
        self.models_dict = {}
    
    def select_models(self):
        classification_models = {
            'LogisticRegression': (LogisticRegression(), {'C': [0.1, 1, 10]}),
            'DecisionTreeClassifier': (DecisionTreeClassifier(), {'max_depth': [5, 10]}),
            'RandomForestClassifier': (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [5, 10]}),
            'XGBClassifier': (XGBClassifier(eval_metric='mlogloss'), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}),
            'GradientBoostingClassifier': (GradientBoostingClassifier(), {'n_estimators': [50, 100]}),
            'SVC': (SVC(), {'C': [0.1, 1, 10]}),
            'KNeighborsClassifier': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]})
        }

        regression_models = {
            'LinearRegression': (LinearRegression(), {}),
            'DecisionTreeRegressor': (DecisionTreeRegressor(), {'max_depth': [5, 10]}),
            'RandomForestRegressor': (RandomForestRegressor(), {'n_estimators': [50, 100], 'max_depth': [5, 10]}),
            'XGBRegressor': (XGBRegressor(), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}),
            'GradientBoostingRegressor': (GradientBoostingRegressor(), {'n_estimators': [50, 100]}),
            'SVR': (SVR(), {'C': [0.1, 1, 10]}),
            'KNeighborsRegressor': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]}),
            'Lasso': (Lasso(), {'alpha': [0.01, 0.1, 1]}),
            'Ridge': (Ridge(), {'alpha': [0.01, 0.1, 1]}),
            'ElasticNet': (ElasticNet(), {'alpha': [0.01, 0.1, 1]})
        }

        all_models = classification_models if self.task_type == 'classification' else regression_models

        if self.model_names == 'all':
            self.models = [(k, v[0], v[1]) for k, v in all_models.items()]
        else:
            self.models = [(name, all_models[name][0], all_models[name][1]) for name in self.model_names if name in all_models]

    def tune_and_train(self, X_train, y_train, X_test, y_test):
        total = len(self.models)
        
        for i, (model_name, model, param_grid) in enumerate(self.models):
            with st.spinner(f"🔍 Tuning {model_name}..."):
                try:
                    if len(X_train) <= 100000:
                        search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring=self._get_scoring())
                    else:
                        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, n_jobs=-1, 
                                                 scoring=self._get_scoring(), random_state=42)

                    search.fit(X_train, y_train)
                    score = search.score(X_test, y_test)
                    self.model_scores[model_name] = score
                    self.models_dict[model_name] = search.best_estimator_

                    if self.task_type == 'classification':
                        if score > self.best_score:
                            self.best_score = score
                            self.best_model = search.best_estimator_
                    else:
                        if score < self.best_score:
                            self.best_score = score
                            self.best_model = search.best_estimator_

                    st.success(f"✅ {model_name} Score: {score:.4f}")
                except Exception as e:
                    st.error(f"❌ Error with model {model_name}: {str(e)}")

    def _get_scoring(self):
        return 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'


class PipelineSaver:
    def __init__(self, path='saved_pipeline'):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def save(self, model, preprocessor):
        model_path = os.path.join(self.path, 'best_model.pkl')
        scaler_path = os.path.join(self.path, 'scaler.pkl')
        
        joblib.dump(model, model_path)
        joblib.dump(preprocessor.scaler, scaler_path)
        
        for name, enc in preprocessor.encoders.items():
            enc_path = os.path.join(self.path, f'{name}_encoder.pkl')
            joblib.dump(enc, enc_path)
            
        if preprocessor.smote:
            smote_path = os.path.join(self.path, 'smote.pkl')
            joblib.dump(preprocessor.smote, smote_path)

        zip_path = os.path.join(self.path, 'automl_artifacts.zip')
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for root, _, files in os.walk(self.path):
                for file in files:
                    if file.endswith('.pkl'):
                        file_path = os.path.join(root, file)
                        zf.write(file_path, arcname=file)
        
        return zip_path
