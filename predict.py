from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("pneumonia-detection-model.h5")

# Load and preprocess the input image
img_path = "sample_image.jpeg"  # Replace with your image filename if different
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
confidence = prediction[0][0]

# Output result
if confidence > 0.5:
    print(f"Prediction: PNEUMONIA ({confidence * 100:.2f}% confidence)")
else:
    print(f"Prediction: NORMAL ({(1 - confidence) * 100:.2f}% confidence)")
