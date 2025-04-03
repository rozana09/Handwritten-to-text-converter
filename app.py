from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from inferenceModel import ImageToWordModel
from mltu.configs import BaseModelConfigs

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model configs and initialize the model
configs = BaseModelConfigs.load("Models/03_handwriting_recognition/202301111911/configs.yaml")
model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Read the image and run prediction
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400

        prediction_text = model.predict(image)
        return jsonify({"text": prediction_text})

    return jsonify({"error": "Unexpected error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
