import sys
import os 
import pandas as pd
from src.exception import CustomException
from src.utils import load_obj

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            print("before loading ")
            model = load_obj(file_path = model_path)
            preprocessor = load_obj(file_path = model_path)
            print("after loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
            
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __int__(self,Brand:str ,Model:str ,Color: str ,Rating:int ,Original_Price : int , Memory :int ,Storage: int):
        self.brand = Brand
        self.model = Model
        self.color = Color
        self.rating = Rating
        self.orginal_price = Original_Price
        self.memory = Memory
        self.storage = Storage
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "brand" :[self.brand],
                "model" : [self.model],
                "color" : [self.color],
                "rating" : [self.rating],
                "original_price" : [self.orginal_price],
                "memory" : [self.memory],
                "storage" : [self.storage]
                
                }
        
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e ,sys)