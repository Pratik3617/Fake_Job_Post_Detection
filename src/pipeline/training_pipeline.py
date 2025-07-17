from src.components.data_validation import DataValidation
from src.exception import MyException
from src.logger import logging
import sys

from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()

    
    def run_pipeline(self) -> None:
        try:
            # data ingestion
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Exited the data_ingestion method of TrainingPipeline class")

            # data validation
            data_validation = DataValidation(data_ingestion_artifact, self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Exited the data_validation method of TrainingPipeline class")
        except Exception as e:
            raise MyException(e,sys)