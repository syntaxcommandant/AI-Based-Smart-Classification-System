from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
app = Flask(__name__)

model = load_model("model/waste_classifier.keras")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    if file:
        filepath = "uploads/" + file.filename
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        print("Prediction value:",prediction)

        if prediction[0][0] > 0.5:
            result = "Wet Waste"
        else:
            result = "Dry Waste"
    eco_guide = {
    "Wet Waste": {
        "bin": "Green Bin",
        "compost": "Yes",
        "recycle": "No",
        "tip": "Kitchen waste can be composted to make natural fertilizer."
    },

    "Dry Waste": {
        "bin": "Blue Bin",
        "compost": "No",
        "recycle": "Yes",
        "tip": "Clean recyclable items before placing them in the recycling bin."
    }
}
    guide = eco_guide[result]


    return render_template(
    "index.html",
    prediction=result,
    guide=guide
)
if __name__ == "__main__":
    app.run(debug=True)

