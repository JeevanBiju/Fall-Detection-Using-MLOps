import sys
import os

from pathlib import Path

# Force correct project root path (3 levels up)
CURRENT_DIR = Path(__file__).resolve().parents[3]
SRC_PATH = CURRENT_DIR / "src"
sys.path.insert(0, str(SRC_PATH))
# Add the `src` directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ChestCancerClassifier.config.configuration import ConfigurationManager
from ChestCancerClassifier.components.model_trainer import Training
from ChestCancerClassifier import logger

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
