import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

X,y = datasets.load_digits(n_class=10,return_X_y=(True),as_frame=(True))

print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

xgb = XGBClassifier()
xgb.fit(X_train,y_train)

print(classification_report(xgb.predict(X_test),y_test))

import joblib
joblib.dump(xgb,"/home/mahmoud/Documents/ML/Number recognizer/XGBoost.pkl")


print(xgb.predict(X_test))