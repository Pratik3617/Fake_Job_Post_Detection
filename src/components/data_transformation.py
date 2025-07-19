import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

            self.numerical_columns = self._schema_config.get('numerical_columns', [])
            self.categorical_columns = self._schema_config.get('categorical_columns', [])
            self.drop_columns = self._schema_config.get('drop_columns', [])

            self.text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
            self.cat_cols = [col for col in self.categorical_columns
                             if col not in self.text_cols and col not in ['department', 'salary_range']]
            self.num_cols = [col for col in self.numerical_columns if col not in ['job_id', TARGET_COLUMN]]

            self.drop_cols = []  # Will be updated dynamically

            self.unknown_fill_cols = self.cat_cols.copy()
            self.text_fill_cols = ['company_profile', 'requirements', 'benefits']

            logging.info(f"Schema loaded - Text: {self.text_cols} | Categorical: {self.cat_cols} | "
                         f"Numerical: {self.num_cols}")
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def _drop_low_value_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dynamically drop columns with:
        - More than 50% missing values
        - Weak differentiation between target classes
        """
        try:
            logging.info("Evaluating columns for dropping (nulls > 50% & weak class separation)")
            self.drop_cols = []

            for col in df.columns:
                null_ratio = df[col].isnull().mean()
                if null_ratio > 0.5 and col != TARGET_COLUMN:
                    if col not in self.text_cols + self.cat_cols + self.num_cols:
                        continue  # Only check relevant columns

                    try:
                        dist = df.groupby(TARGET_COLUMN)[col].value_counts(normalize=True).unstack().fillna(0)
                        max_diff = dist.max() - dist.min()
                        avg_diff = max_diff.mean() if hasattr(max_diff, 'mean') else max_diff

                        if avg_diff < 0.2:
                            logging.info(f"Dropping column '{col}' (nulls={null_ratio:.2f}, avg diff={avg_diff:.2f})")
                            self.drop_cols.append(col)

                    except Exception as ex:
                        logging.warning(f"Could not evaluate column '{col}' for target impact: {ex}")

            df = df.drop(columns=self.drop_cols, errors='ignore')
            logging.info(f"Final dropped columns: {self.drop_cols}")
            return df
        except Exception as e:
            raise MyException(e, sys)

    def _handle_missing_values(self, df: pd.DataFrame, drop_critical: bool = False) -> pd.DataFrame:
        try:
            logging.info("Handling missing values")

            for col in self.unknown_fill_cols:
                if col in df.columns and df[col].isnull().sum() > 0:
                    df[col].fillna('Unknown', inplace=True)

            for col in self.text_fill_cols:
                if col in df.columns and df[col].isnull().sum() > 0:
                    df[col].fillna('', inplace=True)

            if 'title' in df.columns and df['title'].isnull().sum() > 0:
                df['title'].fillna('No Title', inplace=True)

            if drop_critical and 'description' in df.columns:
                before = len(df)
                df.dropna(subset=['description'], inplace=True)
                logging.info(f"Dropped {before - len(df)} rows with missing 'description'")

            return df
        except Exception as e:
            raise MyException(e, sys)

    def _validate_columns(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        try:
            logging.info(f"Validating columns for {data_type} data")

            expected_cols = self.text_cols + self.cat_cols + self.num_cols
            if TARGET_COLUMN in df.columns:
                expected_cols.append(TARGET_COLUMN)

            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                logging.warning(f"Missing columns in {data_type} data: {missing_cols}")
                self.text_cols = [col for col in self.text_cols if col in df.columns]
                self.cat_cols = [col for col in self.cat_cols if col in df.columns]
                self.num_cols = [col for col in self.num_cols if col in df.columns]

            for col in self.num_cols:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            if not self.text_cols:
                raise MyException("No text columns available after validation.", sys)

            return df
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            logging.info("Creating data transformer object")
            transformers = []

            for col in self.text_cols:
                transformers.append((
                    f"text_{col}",
                    TfidfVectorizer(
                        stop_words='english',
                        max_features=300,
                        lowercase=True,
                        strip_accents='unicode',
                        ngram_range=(1, 2)
                    ),
                    col
                ))

            if self.cat_cols:
                cat_pipeline = Pipeline([
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True, drop='first'))
                ])
                transformers.append(('cat', cat_pipeline, self.cat_cols))

            if self.num_cols:
                transformers.append(('num', 'passthrough', self.num_cols))

            if not transformers:
                raise MyException("No valid transformers configured", sys)

            preprocessor = ColumnTransformer(
                transformers=transformers,
                verbose=True,
                remainder='drop'
            )

            return preprocessor
        except Exception as e:
            logging.exception("Failed to create transformer object")
            raise MyException(e, sys)

    def _prepare_data_for_transformation(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        try:
            logging.info(f"Preparing {data_type} data")
            df = self._drop_low_value_columns(df)
            df = self._handle_missing_values(df, drop_critical=(data_type == "train"))
            df = self._validate_columns(df, data_type)
            return df
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            logging.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

            train_df = self._prepare_data_for_transformation(train_df, "train")
            test_df = self._prepare_data_for_transformation(test_df, "test")

            X_train, y_train = train_df.drop(columns=[TARGET_COLUMN]), train_df[TARGET_COLUMN]
            X_test, y_test = test_df.drop(columns=[TARGET_COLUMN]), test_df[TARGET_COLUMN]

            preprocessor = self.get_data_transformer_object()

            logging.info("Fitting transformer on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)

            logging.info("Transforming test data")
            X_test_transformed = preprocessor.transform(X_test)

            if hasattr(X_train_transformed, 'toarray'):
                X_train_transformed = X_train_transformed.toarray()
            if hasattr(X_test_transformed, 'toarray'):
                X_test_transformed = X_test_transformed.toarray()

            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            logging.info("Data Transformation Completed")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            logging.exception("Exception in initiate_data_transformation")
            raise MyException(e, sys)
