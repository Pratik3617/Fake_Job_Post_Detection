from src.exception import MyException
from src.logger import logging
import sys

from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    
    def run_pipeline(self) -> None:
        try:
            # data ingestion
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
        except Exception as e:
            raise MyException(e,sys)