import pandas as pd
import click
from Preprocess.preprocess import Preprocess
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier
from Model.model import Models


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
def main(data_path):

    dataframe = pd.read_csv(data_path)
    X_train = dataframe.drop(["id", "CustomerId", "Surname", "Exited"], axis=1)
    y_train = dataframe["Exited"]
    
    preprocessing = Preprocess(dataframe)
    numeric_processor = Pipeline(
        steps=[("Impute_missing_data", preprocessing.Impute_missing_data()),
               ("Impute_outlier_data", preprocessing.Impute_outlier_data()),
               ("feature_engineering", preprocessing.feature_engineering())]
    )

    categoric_processor = Pipeline(
        steps=[("Impute_missing_data", preprocessing.Impute_missing_data()),
               ("Impute_outlier_data", preprocessing.Impute_outlier_data()),
               ("feature_engineering", preprocessing.feature_engineering()),
               ("ordinalencoding", preprocessing.ordinalencoding()),
               ("onehotencoding", preprocessing.onehotencoding())]
    )

    preprocessor_pipeline = ColumnTransformer(
        [("categorical", categoric_processor),
        ("numerical", numeric_processor),
        ("standardscaler", preprocessing.normalization())]
    )

    print(preprocessor_pipeline)


    ensemble_model = VotingClassifier(
        estimators=[
            "Models", Models()
            ("XgBoost", Models.xgboost_model()),
            ("LightGBM", Models.lgbm_model()),
            ("CatBoost", Models.catboost_model())
        ],
        voting="soft"
    )

    full_pipeline = Pipeline([
        ("preprocessing", preprocessor_pipeline),  # Ön işleme pipeline
        ("ensemble", ensemble_model)  # Ensemble model
    ])

    #preprocessing_pipeline = make_pipeline(full_pipeline)



if __name__ == '__main__':
    main()
