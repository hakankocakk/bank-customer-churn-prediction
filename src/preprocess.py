
import pandas as pd 
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from imblearn.under_sampling import NearMiss
from sklearn.base import BaseEstimator, TransformerMixin



class Preprocess():

    def __init__(self, X, y):
        self.dataframe = X
        self.labels = y
        self.cat_cols = []
        self.num_cols = []
        self.cat_but_car = []

    def create_col_type(self, dataframe, threshold_cat = 4, threshold_car = 20):
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes =='O']
        
        cat_but_car = [col for col in dataframe.columns if ((dataframe[col].dtypes =='O') 
                                                            and (dataframe[col].nunique() > threshold_car))]
        num_but_cat = [col for col in dataframe.columns if ((dataframe[col].dtypes !='O') 
                                                            and (dataframe[col].nunique() <= threshold_cat))]
        cat_cols = [col for col in cat_cols if col not in cat_but_car]
        cat_cols = cat_cols + num_but_cat
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes !='O']
        num_cols = [col for col in num_cols if col not in num_but_cat]
        #num_cols = [col for col in num_cols if col not in ['id', 'CustomerId']]
        #cat_cols = [col for col in cat_cols if col not in ["Exited", "Surname"]]
        cat_cols = [col for col in cat_cols if col not in ["Exited"]]
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.cat_but_car = cat_but_car
        return self.cat_cols, self.num_cols, self.cat_but_car
    
    
    def Impute_missing_num_data(self, dataframe):
        self.cat_cols, self.num_cols, self.cat_but_car = self.create_col_type(dataframe)
        for col in self.num_cols:
            dataframe.loc[dataframe[col].isnull(), col] = dataframe[col].median()
        return dataframe


    def Impute_missing_cat_data(self, dataframe):
        self.cat_cols, self.num_cols, self.cat_but_car = self.create_col_type(dataframe)
        for col in self.cat_cols:
            dataframe.loc[dataframe[col].isnull(), col] = "Unknown"
        return dataframe

    def Impute_outlier_data(self, dataframe, q1=0.15, q3=0.85):
        self.cat_cols, self.num_cols, self.cat_but_car = self.create_col_type(dataframe)
        for col in self.num_cols:
            dataframe[col] = dataframe[col].astype(float)

            quartile_1 = dataframe[col].quantile(q1)
            quartile_3 = dataframe[col].quantile(q3)
            inter_quantile = quartile_3 - quartile_1

            low_limit = quartile_1 - 1.5 * inter_quantile
            upp_limit = quartile_3 + 1.5 * inter_quantile

            dataframe.loc[(dataframe[col] < low_limit), col] = low_limit
            dataframe.loc[(dataframe[col] > upp_limit), col] = upp_limit
        
        return dataframe

    def feature_engineering(self, dataframe):
        dataframe.loc[(dataframe["HasCrCard"]==1) & (dataframe["IsActiveMember"]==1), 
                           "NEW_ACTİVE_CARD"] = "active_card"
        dataframe.loc[(dataframe["HasCrCard"]==1) & (dataframe["IsActiveMember"]==0), 
                           "NEW_ACTİVE_CARD"] = "inactive_card"
        dataframe.loc[(dataframe["HasCrCard"]==0) & (dataframe["IsActiveMember"]==1), 
                           "NEW_ACTİVE_CARD"] = "active_notcard"
        dataframe.loc[(dataframe["HasCrCard"]==0) & (dataframe["IsActiveMember"]==0), 
                           "NEW_ACTİVE_CARD"] = "inactive_notcard"
        
        dataframe.loc[(dataframe["Age"] >= 18) & (dataframe["Age"] < 30), 
                           "NEW_AGE_CAT"] = "young"
        dataframe.loc[(dataframe["Age"] >= 30) & (dataframe["Age"] < 50), 
                           "NEW_AGE_CAT"] = "mature"
        dataframe.loc[dataframe["Age"] >= 50, 
                           "NEW_AGE_CAT"] = "senior"
        
        dataframe.loc[(dataframe["CreditScore"] < 489), 
                           "NEW_CREDİTSCORE_CAT"] = "very_risky"
        dataframe.loc[(dataframe["CreditScore"] >= 489) & (dataframe["CreditScore"] < 599), 
                           "NEW_CREDİTSCORE_CAT"] = "risky"
        dataframe.loc[(dataframe["CreditScore"] >= 599) & (dataframe["CreditScore"] < 704), 
                           "NEW_CREDİTSCORE_CAT"] = "normal"
        dataframe.loc[(dataframe["CreditScore"] >= 704) & (dataframe["CreditScore"] < 812), 
                           "NEW_CREDİTSCORE_CAT"] = "not_risky"
        dataframe.loc[(dataframe["CreditScore"] >= 812), 
                           "NEW_CREDİTSCORE_CAT"] = "very_not_risky"
        
        dataframe.loc[dataframe["Balance"] >= dataframe["EstimatedSalary"], 
                           "NEW_IS_INVESTOR"] = "investor"
        dataframe.loc[dataframe["Balance"] < dataframe["EstimatedSalary"], 
                           "NEW_IS_INVESTOR"] = "not_investor"
        
        dataframe.loc[(dataframe["Geography"] == "France") & (dataframe["NEW_IS_INVESTOR"] == "investor"), 
                           "NEW_IS_GEOGRAPHY_INVESTOR"] = "french_investor"
        dataframe.loc[(dataframe["Geography"] == "Spain") & (dataframe["NEW_IS_INVESTOR"] == "investor"), 
                           "NEW_IS_GEOGRAPHY_INVESTOR"] = "spanish_investor"
        dataframe.loc[(dataframe["Geography"] == "Germany") & (dataframe["NEW_IS_INVESTOR"] == "investor"), 
                           "NEW_IS_GEOGRAPHY_INVESTOR"] = "german_investor"
        dataframe.loc[(dataframe["Geography"] == "France") & (dataframe["NEW_IS_INVESTOR"] == "not_investor"), 
                           "NEW_IS_GEOGRAPHY_INVESTOR"] = "french_not_investor"
        dataframe.loc[(dataframe["Geography"] == "Spain") & (dataframe["NEW_IS_INVESTOR"] == "not_investor"), 
                           "NEW_IS_GEOGRAPHY_INVESTOR"] = "spanish_not_investor"
        dataframe.loc[(dataframe["Geography"] == "Germany") & (dataframe["NEW_IS_INVESTOR"] == "not_investor"), 
                           "NEW_IS_GEOGRAPHY_INVESTOR"] = "german_not_investor"
        
        return dataframe
    

    def ordinalencoding_num(self, dataframe, train=True):
        age_cat_size = ['young', 'mature', 'senior']
        credi_score_size = ['very_risky', 'risky', 'normal', 'not_risky', 'very_not_risky']

        enc = OrdinalEncoder(categories=[ age_cat_size, credi_score_size])
        columns_to_encode = ["NEW_AGE_CAT", "NEW_CREDİTSCORE_CAT"]
        if train:
            dataframe[columns_to_encode] = enc.fit_transform(dataframe[columns_to_encode])
            joblib.dump(enc, 'data/ordinal_encoder.pkl')
        else:
            loaded_encoder = joblib.load('data/ordinal_encoder.pkl')
            dataframe[columns_to_encode] = loaded_encoder.transform(dataframe[columns_to_encode])

        return dataframe


    def onehotencoding(self, dataframe, train=True):
        self.cat_cols, self.num_cols, self.cat_but_car = self.create_col_type(dataframe)
        one_hot_cat_cols = [col for col in self.cat_cols if col not in ["NEW_AGE_CAT", "NEW_CREDİTSCORE_CAT"]]
        

        if train:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            encoded_cols = ohe.fit_transform(dataframe[one_hot_cat_cols])
            joblib.dump(ohe, 'data/one_hot_encoder.pkl')
            new_columns = ohe.get_feature_names_out(one_hot_cat_cols)
            encoded_df = pd.DataFrame(encoded_cols, columns=new_columns, index=dataframe.index)
            dataframe = pd.concat([dataframe, encoded_df], axis=1)
            dataframe.drop(columns=one_hot_cat_cols, inplace=True)
        else:
            loaded_ohe = joblib.load('data/one_hot_encoder.pkl')
            encoded_test_data = loaded_ohe.transform(dataframe[one_hot_cat_cols])
            new_columns = loaded_ohe.get_feature_names_out(one_hot_cat_cols)
            encoded_test_df = pd.DataFrame(encoded_test_data, columns=new_columns, index=dataframe.index)
            dataframe = pd.concat([dataframe.reset_index(drop=True), encoded_test_df], axis=1)
            dataframe.drop(columns=one_hot_cat_cols, inplace=True)
        
        return dataframe
    

    def normalization(self, dataframe, train=True):
        self.cat_cols, self.num_cols, self.cat_but_car = self.create_col_type(dataframe)
        
        if train:
            scaler = StandardScaler()
            dataframe[self.num_cols] = scaler.fit_transform(dataframe[self.num_cols])
            joblib.dump(scaler, 'data/standardscaler.pkl')
        else:
            loaded_scaler = joblib.load('data/standardscaler.pkl')
            dataframe[self.num_cols] = loaded_scaler.transform(dataframe[self.num_cols])

        return dataframe

    
