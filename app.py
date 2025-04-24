from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load pre-trained models (make sure these files are in the same folder)
pneumonia_model = load_model("pneumonia-detection-model.h5")
cancer_model = load_model("cancer-detection-model.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    confidence = None
    image_url = ""
    disease_type = ""

    name = age = mobile = email = ""

    try:
        if request.method == "POST":
            # Get patient details
            name = request.form["name"]
            age = request.form["age"]
            mobile = request.form["mobile"]
            email = request.form["email"]
            disease_type = request.form["disease"]
            file = request.files["file"]

            # Save uploaded file
            os.makedirs("static/uploaded", exist_ok=True)
            filename = file.filename
            filepath = os.path.join("static", "uploaded", filename)
            file.save(filepath)
            image_url = filepath

            # Preprocess image
            img = image.load_img(filepath, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            if disease_type == "pneumonia":
                prob = float(pneumonia_model.predict(img_array)[0][0])
                confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
                prediction = "PNEUMONIA" if prob > 0.5 else "NORMAL"

            elif disease_type == "cancer":
                prob = float(cancer_model.predict(img_array)[0][0])
                confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
                prediction = "CANCER" if prob > 0.5 else "NORMAL"

    except Exception as e:
        prediction = f"Error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        disease_type=disease_type,
        name=name,
        age=age,
        mobile=mobile,
        email=email
    )

if __name__ == "__main__":
    app.run(debug=True)
