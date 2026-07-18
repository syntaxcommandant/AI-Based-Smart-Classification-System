from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("model/waste_classifier.keras")
img = image.load_img("broken-egg-shells.jpg", target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0
prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("Wet Waste")
else:
    print("Dry Waste")


