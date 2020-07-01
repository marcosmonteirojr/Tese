import xgboost
import numpy as np

from scipy.stats import uniform, randint

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
X,y=load_breast_cancer(return_X_y=True)
print(y)

xgb_model = xgboost.XGBClassifier\
    (objective="binary:logistic", random_state=42)
xgb_model.fit(X, y)