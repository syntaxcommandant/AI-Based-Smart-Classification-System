from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

app = Flask(__name__)

model = load_model("model/waste_classifier.keras")

classes = ["Cardboard", "E-Waste", "Glass", "Metal", "Organic", "Paper", "Plastic"]

eco_guide = {
    "Cardboard": {
        "bin": "Blue Bin",
        "compost": "No",
        "recycle": "Yes",
        "tip": "Flatten cardboard boxes before recycling to save space."
    },
    "E-Waste": {
        "bin": "Special E-Waste Collection Point",
        "compost": "No",
        "recycle": "Yes",
        "tip": "Never throw e-waste in regular bins; drop it at authorized e-waste centers."
    },
    "Glass": {
        "bin": "Blue Bin",
        "compost": "No",
        "recycle": "Yes",
        "tip": "Rinse glass containers before recycling."
    },
    "Metal": {
        "bin": "Blue Bin",
        "compost": "No",
        "recycle": "Yes",
        "tip": "Crush cans to save recycling bin space."
    },
    "Organic": {
        "bin": "Green Bin",
        "compost": "Yes",
        "recycle": "No",
        "tip": "Kitchen waste can be composted to make natural fertilizer."
    },
    "Paper": {
        "bin": "Blue Bin",
        "compost": "No",
        "recycle": "Yes",
        "tip": "Keep paper dry and clean for better recycling quality."
    },
    "Plastic": {
        "bin": "Blue Bin",
        "compost": "No",
        "recycle": "Yes",
        "tip": "Clean recyclable plastic items before placing them in the recycling bin."
    }
}


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
        img_array = preprocess_input(img_array)   
        prediction = model.predict(img_array)
        print("Prediction value:", prediction)

        index = np.argmax(prediction[0])
        result = classes[index]
        confidence = float(prediction[0][index])

        guide = eco_guide[result]

        return render_template(
            "index.html",
            prediction=result,
            confidence=round(confidence * 100, 2),
            guide=guide
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
