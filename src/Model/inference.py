import pandas as pd
import os
from ..Preprocess.preprocess import Preprocess
from sklearn.ensemble import VotingClassifier
import joblib
import click



def load_model():

    model_path = os.path.join(os.path.join(os.path.dirname(__file__), "models"))
    ensemble_model = joblib.load(os.path.join(model_path, "ensemble_model.pkl"))

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




