import numpy as np
import pandas as pd
import os 
import sys
import pickle 
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn


def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj :
            pickle.dump(obj ,file_obj)
            
    except Exception as e :
        raise CustomException(e,sys)
    
def  evaluate_model(X_train, y_train,X_test,y_test,models,param):
    try:
        report= {}
        trained_models= {}
        mlflow.sklearn.autolog(log_models=True ,max_tuning_runs=None)
        
        for model_name , model in models.items():
            para = param[model_name]
            
            gs = GridSearchCV(model,para,cv=3 , scoring="r2")
            gs.fit(X_train,y_train)
             
            best_model = gs.best_estimator_
            
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            mlflow.log_metric("test_r2_score", test_model_score)
            report[model_name] = test_model_score
            trained_models[model_name] = best_model
        
        return report , trained_models
       
    except Exception as e:
        raise CustomException(e,sys)
    
def load_obj(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e ,sys)
    