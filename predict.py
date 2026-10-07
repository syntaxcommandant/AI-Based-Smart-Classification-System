import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model("model/waste_classifier.keras")

classes = [
    "Cardboard",
    "E-Waste",
    "Glass",
    "Metal",
    "Organic",
    "Paper",
    "Plastic"
]

import os

img_path = "test_img.jpg" if os.path.exists("test_img.jpg") else "dataset/Glass/white-glass471.jpg"
img = image.load_img(
    img_path,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

prediction = model.predict(img_array)
print("Prediction:", prediction)

index = np.argmax(prediction)

print("Predicted class:", classes[index])
print("Confidence:", prediction[0][index])

