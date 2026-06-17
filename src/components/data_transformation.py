import pandas as pd
import re
import numpy as np
import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj__file_path = os.join.path("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def clean_memory_storage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies your custom regex engine to clean Memory and Storage."""
        logging.info("Applying Regex cleaning logic on Memory and Storage features.")
        
        def extract_and_convert(val):
            val = str(val).lower().strip()
            match = re.search(r"(\d+\.?\d*)\s*(gb|mb)", val)
            if match:
                number = float(match.group(1))
                unit = match.group(2)
                return number / 1024 if unit == 'mb' else number
            
            only_number = re.search(r"(\d+\.?\d*)", val)
            return float(only_number.group(1)) if only_number else 0.0

        for col in ['Memory', 'Storage']:
            df[col] = df[col].convert_dtypes().astype(str).str.lower().str.split()
            df[f'{col}_Cleaned'] = df[col].apply(extract_and_convert)
            
        return df
    
    def filter_outliers_and_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters valid ranges and drops invalid rows entirely."""
        logging.info("Filtering valid ranges and removing unauthentic data rows.")
        
        # Enforce range limits
        df['Storage_Cleaned'] = pd.to_numeric(df['Storage_Cleaned'], errors='coerce')
        df['Storage_Cleaned'] = df['Storage_Cleaned'].where(df['Storage_Cleaned'].between(4, 2048), np.nan)
        
        df['Memory_Cleaned'] = pd.to_numeric(df['Memory_Cleaned'], errors='coerce')
        df['Memory_Cleaned'] = df['Memory_Cleaned'].where(df['Memory_Cleaned'].between(0.5, 64), np.nan)
        
        # Drop rows failing the range validation
        df = df.dropna(subset=['Storage_Cleaned', 'Memory_Cleaned'])
        
        # Drop original raw uncleaned source columns
        df = df.drop(columns=["Memory", "Storage"], errors='ignore')
        return df
    
    def get_data_transformer_obj(self):
        
        try:
            numerical_columns =['Memory_Cleaned','Storage_Cleaned','Rating','Original Price'],
            categorical_columns = ['Brand', 'Model', 'Color']
            
            num_pipline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="median")),
                    ("Scalar",StandardScaler())
                ]
            )
            
            cat_pipline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="most frequent")),
                    ("OneHotEncoding",OneHotEncoder())
                ]
            )
            logging.info(f"categorical columns:{categorical_columns}")
            logging.info(f"numerical columns:{numerical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipline",num_pipline,numerical_columns),
                    ("categorical_pipline",cat_pipline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e ,sys)
        
    def initiate_data_transformation(self, train_path ,test_path):
            try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                
                logging.info("reading train and test complete")
            
                train_df = self.clean_memory_storage_columns(train_df)
                train_df = self.filter_outliers_and_ranges(train_df)

                test_df = self.clean_memory_storage_columns(test_df)
                test_df = self.filter_outliers_and_ranges(test_df)
                
                logging.info("Cleaning and Filtering outliers completed")
                
                logging.info("obtaining preprocessing object")
                preprocessing_obj = self.get_data_transformer_obj()
                
                target_column_name = "Selling Price"
                
                input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
                target_feature_train_df = train_df[target_column_name]

                input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
                target_feature_test_df = test_df[target_column_name]
                
                logging.info("Applying preprocessing object on training and testing dataframes.")
                
                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
                
                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr = np.log1p(np.c_[input_feature_test_arr, np.array(target_feature_test_df)])
                
                logging.info("Saving preprocessing object.")
                save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                )

                return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            except Exception as e :
                raise CustomException(e , sys)
                
        
