import sys
import os

from pathlib import Path

# Force correct project root path (3 levels up)
CURRENT_DIR = Path(__file__).resolve().parents[3]
SRC_PATH = CURRENT_DIR / "src"
sys.path.insert(0, str(SRC_PATH))  #  Add correct src path only


from ChestCancerClassifier.config.configuration import ConfigurationManager
from ChestCancerClassifier.components.data_ingestion import DataIngestion
from ChestCancerClassifier import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
