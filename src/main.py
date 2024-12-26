import pandas as pd
import preprocess
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


def main():

    preprocessing = preprocess.Preprocess("C:/Users/Hakan/Desktop/Bank_Customer_Churn_Prediction/data/train.csv")
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

    preprocessing_pipeline = make_pipeline(preprocessor_pipeline)


if __name__ == '__main__':
    main()