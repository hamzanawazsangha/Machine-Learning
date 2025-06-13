# automl_pipeline.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
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
        # Drop rows with all NaNs or unnamed columns
        df.dropna(axis=0, how='all', inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Separate target
        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])

        # Identify categorical and numerical
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Impute missing
        cat_imputer = SimpleImputer(strategy='most_frequent')
        num_imputer = SimpleImputer(strategy='mean')
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        X[num_cols] = num_imputer.fit_transform(X[num_cols])

        # Encode categorical
        if self.task_type != 'unsupervised':
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                self.encoders['target'] = le

        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_encoded = pd.DataFrame(ohe.fit_transform(X[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
        self.encoders['ohe'] = ohe

        X_final = pd.concat([X[num_cols], X_encoded], axis=1)

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
    def __init__(self, task_type, models='all'):
        self.task_type = task_type
        self.models = models
        self.selected_models = []
        self.best_model = None

    def select_models(self):
        if self.task_type == 'classification':
            base_models = [
                ('LogisticRegression', LogisticRegression(max_iter=500)),
                ('RandomForest', RandomForestClassifier()),
                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
            ]
        elif self.task_type == 'regression':
            base_models = [
                ('LinearRegression', LinearRegression()),
                ('RandomForestRegressor', RandomForestRegressor()),
                ('XGBRegressor', XGBRegressor())
            ]
        else:
            raise ValueError("Unsupervised models not implemented yet")

        if self.models == 'all':
            self.selected_models = base_models
        else:
            self.selected_models = [m for m in base_models if m[0] in self.models]

    def tune_and_train(self, X_train, y_train, X_test, y_test):
        best_score = -np.inf
        for name, model in self.selected_models:
            print(f"Tuning model: {name}")
            param_grid = self.get_params(name)
            if len(X_train) <= 100000:
                search = GridSearchCV(model, param_grid, cv=3, scoring=self.get_scoring(), n_jobs=-1)
            else:
                search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3, scoring=self.get_scoring(), n_jobs=-1)

            search.fit(X_train, y_train)
            preds = search.predict(X_test)
            score = self.evaluate(y_test, preds)
            print(f"{name} score: {score}")

            if score > best_score:
                best_score = score
                self.best_model = search.best_estimator_

    def get_params(self, model_name):
        if model_name == 'LogisticRegression':
            return {'C': [0.01, 0.1, 1, 10]}
        elif model_name == 'RandomForest':
            return {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
        elif model_name == 'XGBoost':
            return {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}
        elif model_name == 'LinearRegression':
            return {}
        elif model_name == 'RandomForestRegressor':
            return {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
        elif model_name == 'XGBRegressor':
            return {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}
        return {}

    def get_scoring(self):
        return 'roc_auc' if self.task_type == 'classification' else 'r2'

    def evaluate(self, y_true, y_pred):
        if self.task_type == 'classification':
            return roc_auc_score(y_true, y_pred)
        else:
            return r2_score(y_true, y_pred)


class PipelineSaver:
    def __init__(self, path='saved_pipeline'):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def save(self, model, preprocessor):
        joblib.dump(model, os.path.join(self.path, 'best_model.pkl'))
        joblib.dump(preprocessor.scaler, os.path.join(self.path, 'scaler.pkl'))
        for name, enc in preprocessor.encoders.items():
            joblib.dump(enc, os.path.join(self.path, f'{name}_encoder.pkl'))
        if preprocessor.smote:
            joblib.dump(preprocessor.smote, os.path.join(self.path, 'smote.pkl'))


# Example Usage:
# df = pd.read_csv('your_dataset.csv')
# pre = DataPreprocessor(task_type='classification', target_column='target')
# X, y = pre.preprocess(df)
# X_train, X_test, y_train, y_test = pre.scale_and_split(X, y)
# X_train, y_train = pre.balance_data(X_train, y_train)
#
# trainer = ModelTrainer(task_type='classification')
# trainer.select_models()
# trainer.tune_and_train(X_train, y_train, X_test, y_test)
#
# saver = PipelineSaver()
# saver.save(trainer.best_model, pre)
