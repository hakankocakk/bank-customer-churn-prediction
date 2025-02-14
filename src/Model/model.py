import pandas as pd
import numpy as np
import os
from ..Preprocess.preprocess import Preprocess
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import click
#import cupy as cp




class Models():

    def __init__(self):
        self.save_path = os.path.join(os.path.join(os.path.dirname(__file__), "models"))

    def xgboost_model(self, X_train, y_train):
        sample_weight = np.where(y_train == 1, 3, 0.5)
        xgboost_best_params = {
            "learning_rate": 0.01,
            "max_depth": 5,
            "n_estimators": 1000,
            "colsample_bytree": 0.7,
            "device": "cuda"
            }
        
        xgboost_final = XGBClassifier(**xgboost_best_params).fit(X_train, y_train, sample_weight=sample_weight)
        joblib.dump(xgboost_final, os.path.join(self.save_path, "xgboost_model.pkl"))

        return xgboost_final
    

    def lightgbm_model(self, X_train, y_train):
        sample_weight = np.where(y_train == 1, 3, 0.5)
        lgbm_best_params = {
            "num_leaves": 15,
            "max_depth": -1,
            "learning_rate" : 0.1,
            "n_estimators": 100,
            "verbosity" : -1,
            "device": "gpu"
            }
        lgbm_final = LGBMClassifier(**lgbm_best_params).fit(X_train, y_train, sample_weight = sample_weight)
        joblib.dump(lgbm_final, os.path.join(self.save_path, "lightgbm_model.pkl"))

        return lgbm_final


    def catboost_model(self, X_train, y_train):
        sample_weight = np.where(y_train == 1, 3, 0.5)
        catboost_best_params = {
            "iterations": 200,
            "learning_rate": 0.1,
            "depth" : 4,
            "bagging_temperature": 0.0,
            "verbose" : False,
            "task_type": "GPU"
            }
        catboost_final = CatBoostClassifier(**catboost_best_params).fit(X_train, y_train,  sample_weight = sample_weight)
        joblib.dump(catboost_final, os.path.join(self.save_path, "catboost_model.pkl"))

        return catboost_final
    

    def ensemble_model(self, X_train, y_train):
        sample_weight = np.where(y_train == 1, 3, 0.5)
        ensemble_model = VotingClassifier(
            estimators=[
                ("XgBoost", self.xgboost_model(X_train, y_train)),
                ("LightGBM", self.lightgbm_model(X_train, y_train)),
                ("CatBoost", self.catboost_model(X_train, y_train))
            ],
            voting="soft"
        )
        ensemble_model = ensemble_model.fit(X_train, y_train, sample_weight=sample_weight)
        joblib.dump(ensemble_model, os.path.join(self.save_path, "ensemble_model.pkl"))

        return ensemble_model


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
def train(data_path):
    dataframe = pd.read_csv(data_path)
    X_train = dataframe.drop(["Exited"], axis=1)
    y_train = dataframe["Exited"]
    preprocess = Preprocess(X_train)
    preprocess_data = preprocess.preprocess_pipeline()
    ensemble_model = Models().ensemble_model(preprocess_data, y_train)
 
if __name__ == '__main__':
    train()

            
        

