import logging
import os
import datetime

lOG_FILE = f"{datetime.now().strftime("%m_%D_%Y_%H_%M_%S")}".log
log_path = os.path.join(os.getcwd(),"logs",lOG_FILE)
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path ,lOG_FILE)

logging.basicConfig(
    filename= LOG_FILE_PATH,
    format= "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level= logging.INFO ,
)