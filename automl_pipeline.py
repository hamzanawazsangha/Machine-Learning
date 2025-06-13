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
    
        # Drop rows with missing target
        df.dropna(subset=[self.target_column], inplace=True)
    
        # Separate target AFTER handling NaNs
        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])
    
        # Identify categorical and numerical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
        # Impute missing values
        cat_imputer = SimpleImputer(strategy='most_frequent')
        num_imputer = SimpleImputer(strategy='mean')
        X[cat_cols] = pd.DataFrame(cat_imputer.fit_transform(X[cat_cols]), columns=cat_cols, index=X.index)
        X[num_cols] = pd.DataFrame(num_imputer.fit_transform(X[num_cols]), columns=num_cols, index=X.index)
    
        # Encode target if classification or regression
        if self.task_type != 'unsupervised':
            if y.dtype == 'object':
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y), index=y.index)
                self.encoders['target'] = le
    
        # OneHotEncode categorical features
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_encoded = pd.DataFrame(ohe.fit_transform(X[cat_cols]), columns=ohe.get_feature_names_out(cat_cols), index=X.index)
        self.encoders['ohe'] = ohe
    
        # Combine numerical and encoded categorical features
        X_final = pd.concat([X[num_cols], X_encoded], axis=1)
    
        # Ensure index alignment
        X_final = X_final.loc[y.index]
    
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
    def __init__(self, task_type='classification', models='all'):
        self.task_type = task_type
        self.model_names = models
        self.models = []
        self.best_model = None
        self.best_score = -float('inf') if task_type == 'classification' else float('inf')

    def select_models(self):
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from xgboost import XGBClassifier, XGBRegressor

        all_models = {
            'LogisticRegression': (LogisticRegression(), {'C': [0.1, 1, 10]}),
            'RandomForest': (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [5, 10]}),
            'XGBoost': (XGBClassifier(eval_metric='mlogloss'), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}),
            'LinearRegression': (LinearRegression(), {}),
            'RandomForestRegressor': (RandomForestRegressor(), {'n_estimators': [50, 100], 'max_depth': [5, 10]}),
            'XGBRegressor': (XGBRegressor(), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]})
        }

        if self.model_names == 'all':
            self.models = [v for k, v in all_models.items() if self.task_type in k.lower()]
        else:
            self.models = [all_models[name] for name in self.model_names if name in all_models]

    def tune_and_train(self, X_train, y_train, X_test, y_test):
        total = len(self.models)
        progress = st.progress(0)

        for i, (model_name, model, param_grid) in enumerate(
            [(name, m, p) for (m, p), name in zip(self.models, [name for name in (self.model_names if self.model_names != 'all' else [name for name in ['LogisticRegression','RandomForest','XGBoost','LinearRegression','RandomForestRegressor','XGBRegressor']])])]):
            with st.spinner(f"🔍 Tuning {model_name}..."):
                try:
                    st.write(f"➡️ Starting model: `{model_name}`")
                    if len(X_train) <= 100000:
                        search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring=self._get_scoring())
                    else:
                        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, n_jobs=-1, scoring=self._get_scoring(), random_state=42)

                    search.fit(X_train, y_train)
                    score = search.score(X_test, y_test)

                    st.success(f"✅ {model_name} Score: {score:.4f}")

                    if self.task_type == 'classification':
                        if score > self.best_score:
                            self.best_score = score
                            self.best_model = search.best_estimator_
                    else:
                        if score < self.best_score:
                            self.best_score = score
                            self.best_model = search.best_estimator_

                except Exception as e:
                    st.error(f"❌ Error with model {model_name}: {e}")

            progress.progress((i + 1) / total)

        if self.best_model is not None:
            st.success(f"🏆 Best Model: {type(self.best_model).__name__} with score {self.best_score:.4f}")
        else:
            st.warning("⚠️ No model was successfully trained.")

    def _get_scoring(self):
        return 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'

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
