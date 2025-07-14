import os
import sys
from src.data_access.job_data import JobData
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.logger import logging
from src.exception import MyException

from pandas import DataFrame
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig=DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
        
    def export_data_to_feature_store(self) -> DataFrame:
        """
        exports data from MongoDB to csv file
        """
        try:
            logging.info(f"Exporting data from MongoDB")
            my_data = JobData()
            dataFrame = my_data.export_collection_as_dataframe(self.data_ingestion_config.collection_name)
            logging.info(f"Exported data successfully!!!")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataFrame.to_csv(feature_store_file_path,index=False, header=True)
            return dataFrame
        except Exception as e:
            raise MyException(e,sys)
        
    def split_data(self, dataframe: DataFrame) -> None :
        """
        Split the dataset into train and test set
        """

        logging.info("Splitting Data into train and test set")

        try:
            train, test = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")
            train.to_csv(self.data_ingestion_config.training_file_path,index = False, header=True)
            test.to_csv(self.data_ingestion_config.testing_file_path,index = False, header=True)
            logging.info(f"Exported train and test file path.")

        except Exception as e:
            raise MyException(e,sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion components of the training pipeline

        output: train and test set are returned as the artifacts
        """

        logging.info("Started Data Ingestion Phase....")

        try:
            dataFrame = self.export_data_to_feature_store()
            logging.info("Retrieved data from the mongoDB")

            self.split_data(dataFrame)
            logging.info("Performed train test split on the dataset")

            logging.info("Exited")
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path, test_file_path = self.data_ingestion_config.testing_file_path)
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e,sys)
