from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import os
from flask_cors import CORS, cross_origin
import json
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key'

# Create necessary directories
os.makedirs("Input_Images", exist_ok=True)
os.makedirs("Incorrect_Images", exist_ok=True)

# Load your trained model (ensure the model is in the correct path)
model = tf.keras.models.load_model('artifacts/training/model.h5')  # Replace with your model path

# Define class labels (modify as per your model's classes)
CLASS_LABELS = {0: 'No Fall', 1: 'Fall'}

def decodeImage(imgstring, fileName):
    """Decode a base64 image and save it to a file."""
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)

def preprocess_image(image_path):
    """Preprocess the image for prediction (modify as per your model's requirements)."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Adjust size as per your model's input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def save_incorrect_predictions_to_file(incorrect_predictions, file_path="incorrect_predictions.json"):
    """Save incorrect predictions to a JSON file."""
    with open(file_path, "w") as file:
        json.dump(incorrect_predictions, file, indent=4)
    print(f"Incorrect predictions saved to {file_path}.")

def add_incorrect_prediction(image_data, predicted_label, correct_label, file_path="incorrect_predictions.json"):
    """Add an incorrect prediction to the JSON file."""
    try:
        with open(file_path, "r") as file:
            incorrect_predictions = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        incorrect_predictions = []

    # Save the image data to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"Incorrect_{timestamp}.jpg"
    image_filepath = os.path.join("Incorrect_Images", image_filename)
    decodeImage(image_data, image_filepath)

    new_entry = {
        "image_path": image_filepath,
        "predicted_label": predicted_label,
        "correct_label": correct_label,
        "timestamp": datetime.now().isoformat()
    }

    incorrect_predictions.append(new_entry)
    save_incorrect_predictions_to_file(incorrect_predictions, file_path)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    data = request.get_json()
    image_base64 = data['image']

    # Decode and save the image
    input_image_path = os.path.join("Input_Images", "inputImage.jpg")
    decodeImage(image_base64, input_image_path)

    # Preprocess the image
    img_array = preprocess_image(input_image_path)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))

    result = {
        'predicted_class': predicted_class,
        'predicted_label': CLASS_LABELS.get(predicted_class, "Unknown"),
        'confidence': confidence
    }

    # Return the result
    return jsonify({
        'result': result,
        'predicted_class': predicted_class,
        'message': "Prediction made. Please confirm if this is correct."
    })

@app.route("/feedback", methods=['POST'])
@cross_origin()
def feedbackRoute():
    """Handle user feedback for predictions."""
    file_path = request.form.get('file_path')
    predicted_class = int(request.form.get('predicted_class'))
    feedback = request.form.get('feedback')
    correct_label = request.form.get('correct_label')  # May be None

    print("Received feedback:")
    print(f"File Path: {file_path}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Feedback: {feedback}")
    print(f"Correct Label: {correct_label}")

    if feedback == 'no' and correct_label is not None:
        correct_label = int(correct_label)
        # Store the incorrect prediction
        add_incorrect_prediction(file_path, predicted_class, correct_label)

    return jsonify({'message': 'Thank you for your feedback!'})

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    """Retraining logic based on incorrect predictions."""
    incorrect_file_path = "incorrect.json"
    if not os.path.exists(incorrect_file_path):
        return "No incorrect data available for retraining."

    # Load incorrect predictions
    try:
        with open(incorrect_file_path, "r") as file:
            incorrect_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return "No incorrect data available for retraining."

    if len(incorrect_data) == 0:
        return "No incorrect data available for retraining."

    # Prepare data for retraining
    images = []
    labels = []

    # Extract images and correct labels
    for entry in incorrect_data:
        img_path = entry['image_path']
        correct_label = entry['correct_label']  # already an int
        # Preprocess
        img_array = preprocess_image(img_path)
        images.append(img_array[0])  # img_array is 4D, [0] to get 3D array
        labels.append(correct_label)

    images = np.array(images)
    labels = np.array(labels)
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(CLASS_LABELS))

    # Retraining parameters (adjust as needed)
    learning_rate = 1e-5
    epochs = 3
    batch_size = 4

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Retrain the model on the incorrect samples
    model.fit(images, labels, epochs=epochs, batch_size=batch_size, verbose=1)

    # Save the updated model
    model.save('artifacts/training/model_updated.h5')

    # Clear the incorrect file after successful retraining
    with open(incorrect_file_path, "w") as file:
        file.write("[]")

    return "Retraining completed using incorrect data successfully!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
