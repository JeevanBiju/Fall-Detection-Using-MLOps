import os
import sys
import tensorflow as tf
from zipfile import ZipFile
from pathlib import Path
import urllib.request as request
import time

# Add the `src` directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("Python Path:", sys.path)  # Print Python's import path for debugging

# Ensure the required modules are imported correctly
from box.exceptions import BoxValueError
from ChestCancerClassifier.entity.config_entity import TrainingConfig  # Import the necessary config entity

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        """Loads the base model from the given path"""
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        """Creates training and validation data generators"""
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Saves the model to the specified path"""
        model.save(path)

    def train(self):
        """Trains the model and saves it to the configured path"""
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )


if __name__ == "__main__":
    # Example usage (assuming config is properly passed)
    try:
        # Assume `config` is created from the ConfigurationManager in the pipeline
        config = TrainingConfig(
            root_dir="artifacts",
            trained_model_path=Path("artifacts/model.h5"),
            updated_base_model_path=Path("artifacts/base_model.h5"),
            training_data=Path("data"),
            params_epochs=10,
            params_batch_size=32,
            params_is_augmentation=True,
            params_image_size=[224, 224, 3]
        )

        trainer = Training(config=config)
        trainer.get_base_model()
        trainer.train_valid_generator()
        trainer.train()

    except Exception as e:
        print(f"Error: {e}")
