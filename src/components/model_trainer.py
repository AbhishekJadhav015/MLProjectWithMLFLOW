import os
import sys
import mlflow
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object ,evaluate_model  
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor , RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso ,Ridge 
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig :
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("splitting train and test input data")
            x_train , y_train , x_test , y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
        
            
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            param = {
                "Linear Regression": {},
                
                "Lasso": {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                },
                
                "Ridge": {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                },
                
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                },
                
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2']
                },
                
                "Random Forest Regressor": {  # FIX: Matched naming exactly with models dictionary
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7]
                },
                
                "CatBoosting Regressor": {
                    'depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
                }

            
            model_report , trained_models = evaluate_model(X_train=x_train,X_test=x_test,y_train=y_train,y_test=y_test,models=models,param=param)
            best_model_name = max(model_report ,key =model_report.get)
            best_model_score = model_report[best_model_name]
            
            best_model = trained_models[best_model_name]
           
            if best_model_score < 0.6 :
                raise CustomException("NO best model found")
            logging.info(f"Best model found: {best_model_name} with R2 score of {best_model_score}")            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path ,
                obj= best_model
            )
        
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square
        
        except Exception as e:
            raise CustomException(e ,sys)