import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

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

img = image.load_img(
    "dataset/Glass/white-glass471.jpg",
    target_size=(224, 224)
)

img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print("Prediction:", prediction)

index = np.argmax(prediction)

print("Predicted class:", classes[index])
print("Confidence:", prediction[0][index])

