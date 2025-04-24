from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load both models
pneumonia_model = load_model('pneumonia-detection-model.h5')
cancer_model = load_model('cancer-detection-model.h5')

# Folder to store uploaded files
UPLOAD_FOLDER = 'static/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Prediction function
def predict_disease(img_path, model_type):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    if model_type == "pneumonia":
        prediction = pneumonia_model.predict(img_array)
        confidence = float(prediction[0][0]) * 100
        label = "PNEUMONIA" if confidence > 50 else "NORMAL"

    elif model_type == "cancer":
        prediction = cancer_model.predict(img_array)
        confidence = float(prediction[0][0]) * 100
        label = "CANCER DETECTED" if confidence > 50 else "NO CANCER"

    else:
        label = "INVALID"
        confidence = 0.0

    return label, confidence

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    confidence = None
    disease_type = ""
    image_url = ""

    if request.method == 'POST':
        disease_type = request.form.get('disease')
        file = request.files.get('file')
        if file and disease_type:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction, confidence = predict_disease(filepath, disease_type)
            image_url = filepath

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           disease_type=disease_type,
                           image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
file = request.files["file"]
if file:
    # Ensure upload folder exists
    os.makedirs("static/uploaded", exist_ok=True)

    filename = file.filename
    filepath = os.path.join("static", "uploaded", filename)
    file.save(filepath)

