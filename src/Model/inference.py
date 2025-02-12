import pandas as pd
from ..Preprocess.preprocess import Preprocess
from sklearn.ensemble import VotingClassifier
import joblib
import click



def load_model():
    xgboost_model = joblib.load("models/xgboost_model.pkl")
    lightgbm_model = joblib.load("modelslightgbm_model.pkl")
    catboost_model = joblib.load("models/catboost_model.pkl")

    ensemble_model = VotingClassifier(
        estimators=[
            ("XgBoost", xgboost_model),
            ("LightGBM", lightgbm_model),
            ("CatBoost", catboost_model)
        ],
        voting="soft"
    )

    joblib.dump(ensemble_model, 'C:/Users/Hakan/Documents/GitHub/bank-customer-churn-prediction/src/Model/models/ensemble_model.pkl')
    ensemble_model = joblib.load("C:/Users/Hakan/Documents/GitHub/bank-customer-churn-prediction/src/Model/models/ensemble_model.pkl")

    return ensemble_model

@click.command()
@click.argument('data_path', type=click.Path(exists=True))
def predict(data_path):
    dataframe = pd.read_csv(data_path)
    preprocess = Preprocess(dataframe)
    preprocess_data = preprocess.preprocess_pipeline()
    ensemble_model = load_model()
    predict = ensemble_model.predict(preprocess_data)
    print(predict)


if __name__ == '__main__':
    predict()




