import pandas as pd 
import numpy as np
import joblib
import os
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler




class Preprocess():

    def __init__(self, data):
        self.dataframe  = data.drop(["id", "CustomerId", "Surname"], axis=1)
        self.cat_cols = []
        self.num_cols = []
        self.cat_but_car = []
        self.save_path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "Model/models"))
        #os.makedirs(save_path, exist_ok=True)

    def create_col_type(self, threshold_cat = 11, threshold_car = 20):
        cat_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes =='O']
        
        cat_but_car = [col for col in self.dataframe.columns if ((self.dataframe[col].dtypes =='O') 
                                                            and (self.dataframe[col].nunique() > threshold_car))]
        num_but_cat = [col for col in self.dataframe.columns if ((self.dataframe[col].dtypes !='O') 
                                                            and (self.dataframe[col].nunique() <= threshold_cat))]
        cat_cols = [col for col in cat_cols if col not in cat_but_car]
        cat_cols = cat_cols + num_but_cat
        num_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes !='O']
        num_cols = [col for col in num_cols if col not in num_but_cat]
        num_cols = [col for col in num_cols if col not in ['id', 'CustomerId']]
        cat_cols = [col for col in cat_cols if col not in ["Surname"]]
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.cat_but_car = cat_but_car
        return self.cat_cols, self.num_cols, self.cat_but_car
    
    
    def Impute_missing_data(self, train=True):

        if train == False:
            self.num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
            self.cat_cols = ['Geography', 'Gender', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']

        for col in self.num_cols:
            self.dataframe.loc[self.dataframe[col].isnull(), col] = self.dataframe[col].median()
        for col in self.cat_cols:
            self.dataframe.loc[self.dataframe[col].isnull(), col] = "Unknown"
        return self.dataframe

    def Impute_outlier_data(self, train=True, q1=0.15, q3=0.85):

        if train == False:
            self.num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']

        for col in self.num_cols:
            self.dataframe[col] = self.dataframe[col].astype(float)

            quartile_1 = self.dataframe[col].quantile(q1)
            quartile_3 = self.dataframe[col].quantile(q3)
            inter_quantile = quartile_3 - quartile_1

            low_limit = quartile_1 - 1.5 * inter_quantile
            upp_limit = quartile_3 + 1.5 * inter_quantile

            self.dataframe.loc[(self.dataframe[col] < low_limit), col] = low_limit
            self.dataframe.loc[(self.dataframe[col] > upp_limit), col] = upp_limit
        return self.dataframe

    def feature_engineering(self):

        self.dataframe.loc[(self.dataframe["HasCrCard"]==1) & (self.dataframe["IsActiveMember"]==1), 
                           "NEW_ACTİVE_CARD"] = "active_card"
        self.dataframe.loc[(self.dataframe["HasCrCard"]==1) & (self.dataframe["IsActiveMember"]==0), 
                           "NEW_ACTİVE_CARD"] = "inactive_card"
        self.dataframe.loc[(self.dataframe["HasCrCard"]==0) & (self.dataframe["IsActiveMember"]==1), 
                           "NEW_ACTİVE_CARD"] = "active_notcard"
        self.dataframe.loc[(self.dataframe["HasCrCard"]==0) & (self.dataframe["IsActiveMember"]==0), 
                           "NEW_ACTİVE_CARD"] = "inactive_notcard"

        self.dataframe.loc[self.dataframe["Balance"] >= self.dataframe["EstimatedSalary"], 
                           "NEW_IS_INVESTOR"] = "investor"
        self.dataframe.loc[self.dataframe["Balance"] < self.dataframe["EstimatedSalary"], 
                           "NEW_IS_INVESTOR"] = "not_investor"

        self.dataframe.loc[(self.dataframe["Age"] >= 18) & (self.dataframe["Age"] < 30), 
                           "NEW_AGE_CAT"] = "young"
        self.dataframe.loc[(self.dataframe["Age"] >= 30) & (self.dataframe["Age"] < 50), 
                           "NEW_AGE_CAT"] = "mature"
        self.dataframe.loc[self.dataframe["Age"] >= 50, 
                           "NEW_AGE_CAT"] = "senior"

        self.dataframe.loc[(self.dataframe["Geography"] == "France") & (self.dataframe["NEW_IS_INVESTOR"] == "investor"), 
                           "NEW_IS_GEOGRAPHY_INVESTOR"] = "french_investor"
        self.dataframe.loc[(self.dataframe["Geography"] == "Spain") & (self.dataframe["NEW_IS_INVESTOR"] == "investor"), 
                           "NEW_IS_GEOGRAPHY_INVESTOR"] = "spanish_investor"
        self.dataframe.loc[(self.dataframe["Geography"] == "Germany") & (self.dataframe["NEW_IS_INVESTOR"] == "investor"), 
                           "NEW_IS_GEOGRAPHY_INVESTOR"] = "german_investor"
        self.dataframe.loc[(self.dataframe["Geography"] == "France") & (self.dataframe["NEW_IS_INVESTOR"] == "not_investor"), 
                           "NEW_IS_GEOGRAPHY_INVESTOR"] = "french_not_investor"
        self.dataframe.loc[(self.dataframe["Geography"] == "Spain") & (self.dataframe["NEW_IS_INVESTOR"] == "not_investor"), 
                           "NEW_IS_GEOGRAPHY_INVESTOR"] = "spanish_not_investor"
        self.dataframe.loc[(self.dataframe["Geography"] == "Germany") & (self.dataframe["NEW_IS_INVESTOR"] == "not_investor"), 
                           "NEW_IS_GEOGRAPHY_INVESTOR"] = "german_not_investor"
        
        self.dataframe.loc[(self.dataframe["CreditScore"] < 489), 
                           "NEW_CREDİTSCORE_CAT"] = "very_risky"
        self.dataframe.loc[(self.dataframe["CreditScore"] >= 489) & (self.dataframe["CreditScore"] < 599), 
                           "NEW_CREDİTSCORE_CAT"] = "risky"
        self.dataframe.loc[(self.dataframe["CreditScore"] >= 599) & (self.dataframe["CreditScore"] < 704), 
                           "NEW_CREDİTSCORE_CAT"] = "normal"
        self.dataframe.loc[(self.dataframe["CreditScore"] >= 704) & (self.dataframe["CreditScore"] < 812), 
                           "NEW_CREDİTSCORE_CAT"] = "not_risky"
        self.dataframe.loc[(self.dataframe["CreditScore"] >= 812), 
                           "NEW_CREDİTSCORE_CAT"] = "very_not_risky"
        
        return self.dataframe



    def ordinalencoding(self, train=True):
        investor_size = ['not_investor', 'investor']
        age_cat_size = ['young', 'mature', 'senior']
        credi_score_size = ['very_risky', 'risky', 'normal', 'not_risky', 'very_not_risky']

        enc = OrdinalEncoder(categories=[investor_size, age_cat_size, credi_score_size])
        columns_to_encode = ["NEW_IS_INVESTOR", "NEW_AGE_CAT", "NEW_CREDİTSCORE_CAT"]
        if train:
            self.dataframe[columns_to_encode] = enc.fit_transform(self.dataframe[columns_to_encode])
            joblib.dump(enc, os.path.join(self.save_path, "ordinal_encoder.pkl"))
        else:
            loaded_encoder = joblib.load(os.path.join(self.save_path, "ordinal_encoder.pkl"))
            self.dataframe[columns_to_encode] = loaded_encoder.transform(self.dataframe[columns_to_encode]) 
        return self.dataframe


    def onehotencoding(self, train=True):
        self.cat_cols, self.num_cols, self.cat_but_car = self.create_col_type()
        one_hot_cat_cols = [col for col in self.cat_cols if col not in ["NEW_IS_INVESTOR", "NEW_AGE_CAT", "NEW_CREDİTSCORE_CAT"]]

        if train:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            encoded_cols = ohe.fit_transform(self.dataframe[one_hot_cat_cols])
            joblib.dump(ohe, os.path.join(self.save_path, "onehot_encoder.pkl"))
            new_columns = ohe.get_feature_names_out(one_hot_cat_cols)
            encoded_df = pd.DataFrame(encoded_cols, columns=new_columns, index=self.dataframe.index)
            self.dataframe = pd.concat([self.dataframe, encoded_df], axis=1)
            self.dataframe.drop(columns=one_hot_cat_cols, inplace=True)
        else:
            self.cat_cols = ['Geography', 'Gender', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'NEW_ACTİVE_CARD', 'NEW_IS_GEOGRAPHY_INVESTOR', 'NEW_IS_INVESTOR', 'NEW_AGE_CAT', 'NEW_CREDİTSCORE_CAT']
            one_hot_cat_cols = [col for col in self.cat_cols if col not in ["NEW_IS_INVESTOR", "NEW_AGE_CAT", "NEW_CREDİTSCORE_CAT"]]
            loaded_ohe = joblib.load(os.path.join(self.save_path, "onehot_encoder.pkl"))
            encoded_test_data = loaded_ohe.transform(self.dataframe[one_hot_cat_cols])
            new_columns = loaded_ohe.get_feature_names_out(one_hot_cat_cols)
            encoded_test_df = pd.DataFrame(encoded_test_data, columns=new_columns, index=self.dataframe.index)
            self.dataframe = pd.concat([self.dataframe.reset_index(drop=True), encoded_test_df], axis=1)
            self.dataframe.drop(columns=one_hot_cat_cols, inplace=True)

        return self.dataframe

    def normalization(self, train=True):
        self.cat_cols, self.num_cols, self.cat_but_car = self.create_col_type()

        if train:
            scaler = StandardScaler()
            self.dataframe[self.num_cols] = scaler.fit_transform(self.dataframe[self.num_cols])
            joblib.dump(scaler, os.path.join(self.save_path, "standardscaler.pkl"))
        else:
            self.num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
            loaded_scaler = joblib.load(os.path.join(self.save_path, "standardscaler.pkl"))
            self.dataframe[self.num_cols] = loaded_scaler.transform(self.dataframe[self.num_cols])

        return self.dataframe
    

    def preprocess_pipeline(self, train=True):
        self.create_col_type()
        self.Impute_missing_data(train=train)
        self.Impute_outlier_data(train=train)
        self.feature_engineering()
        self.ordinalencoding(train=train)
        self.onehotencoding(train=train)
        self.normalization(train=train)

        return self.dataframe