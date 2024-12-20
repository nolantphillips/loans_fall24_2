from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd


loans = pd.read_csv('./loan_norm.csv')
strat_train_set, strat_test_set = train_test_split(loans, test_size=0.20, stratify=loans["LoanApproved"], random_state=42)
loans = strat_train_set.drop("LoanApproved", axis=1)
property = strat_train_set["LoanApproved"].copy()

cat_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))

default_num_pipeline = make_pipeline(StandardScaler(), MinMaxScaler((-1,1)))

preprocessing = ColumnTransformer([
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("clf", LogisticRegression()),
])

parameters = [
    {
        'clf': (LogisticRegression(random_state=42),),
        'clf__C': (0.001,0.01,0.1,1,10,100),
        'clf__verbose': [True, False]
    },
    {
        'clf': (RandomForestClassifier(random_state=42),),
        'clf__n_estimators': (10, 30)
    },
    {
        'clf': (RidgeClassifier(random_state=42),),
        'clf__alpha': (0.1, 0.25, 0.5, 0.75, 1),
    },
    {
        'clf': (XGBClassifier(),),
        'clf__n_estimators': (50, 100, 150, 200, 250, 300),
        'clf__max_depth': (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    }
]

grid_search = GridSearchCV(full_pipeline, parameters, cv=3,
                           scoring='f1')

grid_search.fit(loans, property)
loan_predictions = grid_search.best_estimator_.predict(loans)
orig = confusion_matrix(property, loan_predictions)
print("Original Model:", orig)

import dill
with open('xgb_model_v2.pkl', 'wb') as f:
    dill.dump(grid_search.best_estimator_, f)

with open('xgb_model_v2.pkl', 'rb') as f:
    reloaded_model = dill.load(f)

reloaded = confusion_matrix(property, reloaded_model.predict(loans))
print('Reloaded Model:', reloaded)