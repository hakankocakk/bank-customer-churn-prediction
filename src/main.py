import pandas as pd
import preprocess
import joblib
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score


def main():

    dataframe = pd.read_csv("C:/Users/Hakan/bank-customer-churn-prediction/data/train.csv")
    X = dataframe.drop(["id", "CustomerId", "Surname", "Exited"], axis=1)
    y = dataframe["Exited"]

    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=0.15, random_state=42)

    preprocessing = preprocess.Preprocess(X_train, y_train)

    pipeline = Pipeline(
        steps=[
            ("impute_missing_num", FunctionTransformer(preprocessing.Impute_missing_num_data)),
            ("impute_missing_cat", FunctionTransformer(preprocessing.Impute_missing_cat_data)),
            ("impute_outlier", FunctionTransformer(preprocessing.Impute_outlier_data)),
            ("feature_engineering", FunctionTransformer(preprocessing.feature_engineering)),
            ("ordinal_encoding", FunctionTransformer(preprocessing.ordinalencoding_num)),
            ("onehot_encoding", FunctionTransformer(preprocessing.onehotencoding)),
            ("scaler", StandardScaler()), 
            ("model", XGBClassifier())
        ]
    )


    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'src/saved_pipeline.pkl')

    loaded_pipeline = joblib.load('src/saved_pipeline.pkl')
    y_pred = loaded_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.2f}")

if __name__ == '__main__':
    main()