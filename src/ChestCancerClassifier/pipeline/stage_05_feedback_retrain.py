import os
import json
import shutil  # For copying files
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def load_feedback_data(feedback_file, target_size=(224, 224)):
    """Load and preprocess feedback data."""
    if not os.path.exists(feedback_file):
        logging.info(f"No feedback data found at {feedback_file}.")
        return None, None

    with open(feedback_file, "r") as file:
        try:
            incorrect_predictions = json.load(file)
            logging.info(f"Loaded {len(incorrect_predictions)} feedback entries.")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            return None, None

    if not incorrect_predictions:
        logging.info("Feedback data is empty.")
        return None, None

    images, labels = [], []
    for idx, item in enumerate(incorrect_predictions):
        image_path = item.get("image_path")
        correct_label = item.get("correct_label")

        if not image_path or correct_label is None:
            logging.warning(f"Entry {idx} is missing 'image_path' or 'correct_label'. Skipping.")
            continue

        # Handle absolute and relative paths
        if not os.path.isabs(image_path):
            # Assuming image_path is relative to the feedback_file directory
            feedback_dir = os.path.dirname(feedback_file)
            image_path = os.path.join(feedback_dir, image_path)

        if os.path.exists(image_path):
            try:
                image_data = load_img(image_path, target_size=target_size)
                image_data = img_to_array(image_data) / 255.0
                images.append(image_data)
                labels.append(correct_label)
                logging.debug(f"Loaded image {image_path} with label {correct_label}.")
            except Exception as e:
                logging.warning(f"Error loading image {image_path}: {e}")
        else:
            logging.warning(f"Image {image_path} not found. Skipping.")

    if not images:
        logging.info("No valid images found in feedback data.")
        return None, None

    logging.info(f"Successfully loaded {len(images)} images from feedback data.")
    return np.array(images), np.array(labels)

def build_model(base_model_path, num_classes):
    """Build and compile the model."""
    base_model = load_model(base_model_path)
    x = base_model.layers[-4].output  # Example: Adjust as necessary
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze all layers except the last 10
    for layer in model.layers[:-10]:
        layer.trainable = False
    for layer in model.layers[-10:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    logging.info("Model built and compiled successfully.")
    return model

def plot_metrics(history):
    """Plot training and validation metrics."""
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig("artifacts/retraining/training_history.png")
    plt.close()
    logging.info("Training history plot saved to artifacts/retraining/training_history.png")

def retrain_model_with_feedback(base_model_path, feedback_file, save_model_path, num_classes, original_data_dir=None):
    """Retrain the model using feedback data without clearing the feedback file."""
    # Load feedback data
    images, labels = load_feedback_data(feedback_file)
    if images is None or labels is None:
        logging.info("Skipping retraining due to insufficient feedback data.")
        return

    logging.info(f"Number of feedback samples: {len(images)}")
    unique, counts = np.unique(labels, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    logging.info(f"Class distribution: {class_distribution}")

    # Encode labels
    le = LabelEncoder()
    le.fit(labels)  # Assuming labels are integers; adjust if they are strings
    labels_encoded = le.transform(labels)

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight('balanced',
                                         classes=np.arange(num_classes),
                                         y=labels_encoded)
    class_weight_dict = dict(enumerate(class_weights))
    logging.info(f"Class weights: {class_weight_dict}")

    # Augment data
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    augmented_images = []
    augmented_labels = []
    for img, label in zip(images, labels_encoded):
        img = np.expand_dims(img, axis=0)
        aug_iter = datagen.flow(img, batch_size=1)
        for _ in range(5):  # Generate 5 augmented images per original image
            aug_img = next(aug_iter)[0]
            augmented_images.append(aug_img)
            augmented_labels.append(label)
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    logging.info(f"Augmented data size: {len(augmented_images)}")

    # Combine original and augmented data
    X = np.concatenate((images, augmented_images), axis=0)
    y = np.concatenate((labels_encoded, augmented_labels), axis=0)

    # Check the number of unique classes
    num_unique_classes = len(np.unique(y))
    logging.info(f"Number of unique classes after augmentation: {num_unique_classes}")

    if num_unique_classes > 1:
        # Address class imbalance with oversampling
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(
            X.reshape(len(X), -1), y
        )
        X_resampled = X_resampled.reshape(-1, 224, 224, 3)
        logging.info(f"Resampled data size: {len(X_resampled)}")
    else:
        # No oversampling needed as there's only one class
        X_resampled, y_resampled = X, y
        logging.warning("Only one class present in feedback data. Skipping oversampling.")

    # Split into training and validation sets
    if num_unique_classes > 1:
        stratify = y_resampled
    else:
        stratify = None  # Cannot stratify if only one class
    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=stratify
    )

    logging.info(f"Training set size: {len(X_train)}")
    logging.info(f"Validation set size: {len(X_val)}")

    if num_unique_classes > 1:
        # Build and compile the model
        model = build_model(base_model_path, num_classes)

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

        # Train the model
        logging.info("Starting model training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=5,  # Adjust epochs as needed
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weight_dict
        )
        logging.info("Model training completed.")

        # Save the model
        model.save(save_model_path)
        logging.info(f"Updated model saved to {save_model_path}.")

        # Plot metrics
        plot_metrics(history)

if __name__ == "__main__":
    base_model_path = "artifacts/training/model.h5"  # Path to your pre-trained base model
    feedback_file = "incorrect_predictions.json"      # Path to your feedback JSON file
    save_model_path = "artifacts/retraining/updated_model.h5"  # Path to save the updated model
    num_classes = 2  # Replace with your actual number of classes

    retrain_model_with_feedback(base_model_path, feedback_file, save_model_path, num_classes)
