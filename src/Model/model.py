import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import click
#import cupy as cp




class Models():

    def __init__(self, data):
        self.data = data
        self.X_train = self.data.drop(["id", "CustomerId", "Surname", "Exited"], axis=1)
        self.y_train = self.data["Exited"]
        self.sample_weight = np.where(self.y_train == 1, 3, 0.5)
    
    def xgboost_model(self):
        xgboost_best_params = {
            "learning_rate": 0.01,
            "max_depth": 5,
            "n_estimators": 1000,
            "colsample_bytree": 0.7,
            "device": "cuda"
            }
        
        #X_train_gpu = cp.array(X_train)
        #y_train_gpu = cp.array(y_train)
        #class_weights_gpu = cp.array(sample_weight) 
        
        
        xgboost_final = XGBClassifier(**xgboost_best_params).fit(self.X_train, self.y_train, sample_weight=self.sample_weight)
        
        return xgboost_final
    

    def lgbm_model(self):
        lgbm_best_params = {
            "num_leaves": 15,
            "max_depth": -1,
            "learning_rate" : 0.1,
            "n_estimators": 100,
            "verbosity" : -1,
            "device": "gpu"
            }
        
        lgbm_final = LGBMClassifier(**lgbm_best_params).fit(self.X_train, self.y_train, sample_weight = self.sample_weight)

        return lgbm_final

    def catboost_model(self):
        catboost_best_params = {
            "iterations": 200,
            "learning_rate": 0.1,
            "depth" : 4,
            "bagging_temperature": 0.0,
            "verbose" : False,
            "task_type": "GPU"
            }
        
        catboost_final = CatBoostClassifier(**catboost_best_params).fit(self.X_train, self.y_train,  sample_weight = self.sample_weight)

        return catboost_final
        
        

