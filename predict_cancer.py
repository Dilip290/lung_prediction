from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("cancer-detection-model.h5")

# Path to your image
img_path = "sample_image.jpeg"  # Change this if needed

# Load and preprocess image
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
confidence = float(prediction[0][0])

# Output result
if confidence > 0.5:
    print(f"Prediction: CANCER ({confidence * 100:.2f}% confidence)")
else:
    print(f"Prediction: NORMAL ({(1 - confidence) * 100:.2f}% confidence)")
import os
print("Looking for file at:", os.path.abspath("sample_image.jpeg"))

