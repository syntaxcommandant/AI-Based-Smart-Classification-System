from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

import os
import json
import urllib.request
import urllib.error

app = Flask(__name__)

model = load_model("model/waste_classifier.keras")

classes = ["Cardboard", "E-Waste", "Glass", "Metal", "Organic", "Paper", "Plastic"]

default_eco_guide = {
    "Cardboard": {
        "bin": "Blue Bin",
        "compost": "No",
        "recycle": "Yes",
        "tip": "Flatten cardboard boxes before recycling to save space.",
        "carbon_saved": "~0.25 kg CO2 per kg recycled",
        "upcycle_idea": "Use as storage organizer, plant seed starters, or drawer dividers."
    },
    "E-Waste": {
        "bin": "Special E-Waste Collection Point",
        "compost": "No",
        "recycle": "Yes",
        "tip": "Never throw e-waste in regular bins; drop it at authorized e-waste centers.",
        "carbon_saved": "~1.50 kg CO2 equivalent saved",
        "upcycle_idea": "Salvage safe components or donate functioning electronics to local makerspaces."
    },
    "Glass": {
        "bin": "Blue Bin",
        "compost": "No",
        "recycle": "Yes",
        "tip": "Rinse glass containers before recycling.",
        "carbon_saved": "~0.30 kg CO2 per kg recycled",
        "upcycle_idea": "Clean and reuse as drinking jars, spice containers, or mini-planters."
    },
    "Metal": {
        "bin": "Blue Bin",
        "compost": "No",
        "recycle": "Yes",
        "tip": "Crush cans to save recycling bin space.",
        "carbon_saved": "~1.20 kg CO2 per kg recycled (saves up to 95% energy)",
        "upcycle_idea": "Repurpose tin cans into desk pen stands, cutlery holders, or decorative lanterns."
    },
    "Organic": {
        "bin": "Green Bin",
        "compost": "Yes",
        "recycle": "No",
        "tip": "Kitchen waste can be composted to make nutrient-rich natural fertilizer.",
        "carbon_saved": "~0.50 kg methane emissions prevented",
        "upcycle_idea": "Turn into rich organic compost for home plants and kitchen gardens."
    },
    "Paper": {
        "bin": "Blue Bin",
        "compost": "No",
        "recycle": "Yes",
        "tip": "Keep paper dry and clean for better recycling quality.",
        "carbon_saved": "~0.80 kg CO2 per kg recycled",
        "upcycle_idea": "Use for papier-mâché crafts, notebook scratchpads, or compost layering."
    },
    "Plastic": {
        "bin": "Blue Bin",
        "compost": "No",
        "recycle": "Yes",
        "tip": "Clean recyclable plastic items before placing them in the recycling bin.",
        "carbon_saved": "~1.10 kg CO2 per kg recycled",
        "upcycle_idea": "Cut bottles into self-watering plant pots or desk stationery holders."
    }
}


def get_ai_eco_guide(predicted_class, confidence):
    """
    AI Agent Layer:
    Dynamically generates context-aware disposal tips, carbon footprint estimates,
    and DIY upcycling suggestions using LLM API.
    Fails safely to default_eco_guide if API key is not configured or offline.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")

    # If no API key configured, use default enriched guide safely
    if not api_key:
        guide = default_eco_guide.get(predicted_class, default_eco_guide["Plastic"]).copy()
        guide["is_dynamic"] = False
        return guide

    prompt = (
        f"You are an AI Sustainability Agent. A waste item has been classified as '{predicted_class}' "
        f"with {confidence}% confidence. Generate real-time actionable disposal guidance. "
        f"Return ONLY a valid JSON object with these 6 exact string keys:\n"
        f"1. 'bin': colored disposal bin name\n"
        f"2. 'compost': 'Yes' or 'No' with 2-word reason\n"
        f"3. 'recycle': 'Yes' or 'No' with 2-word reason\n"
        f"4. 'tip': 1 concise practical step for proper disposal\n"
        f"5. 'carbon_saved': estimated CO2 / greenhouse gas savings (e.g. ~0.8 kg CO2 saved)\n"
        f"6. 'upcycle_idea': 1 creative DIY upcycle or reuse idea for this waste item\n"
    )

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=4) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
            dynamic_data = json.loads(raw_text)
            dynamic_data["is_dynamic"] = True
            return dynamic_data
    except Exception as e:
        print(f"[EcoAgent] Dynamic generation fallback triggered: {e}")
        guide = default_eco_guide.get(predicted_class, default_eco_guide["Plastic"]).copy()
        guide["is_dynamic"] = False
        return guide


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file and file.filename != "":
        os.makedirs("uploads", exist_ok=True)
        filepath = os.path.join("uploads", file.filename)
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

        # AI Agent layer call
        guide = get_ai_eco_guide(result, round(confidence * 100, 2))

        return render_template(
            "index.html",
            prediction=result,
            confidence=round(confidence * 100, 2),
            guide=guide
        )

    return render_template("index.html")


def get_ai_chat_response(user_msg, detected_class="None", confidence="None"):
    """
    EcoAgent Conversational Assistant:
    Answers user questions about waste disposal, recycling rules, and creative upcycling.
    Uses Gemini API if key is available; otherwise uses intelligent offline Knowledge Engine.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if api_key:
        context_str = f"Scanned item in current session: {detected_class} ({confidence}% confidence)." if detected_class and detected_class != "None" else "No specific item scanned yet."
        prompt = (
            f"You are EcoAgent, a friendly, inspiring, and expert AI Waste Management & Sustainability Assistant. "
            f"Session Context: {context_str}\n"
            f"User Question: '{user_msg}'\n\n"
            f"Instructions:\n"
            f"1. Be direct, clear, and actionable (2-4 concise sentences or bullet points).\n"
            f"2. Specify designated bin colors (Blue = Dry Recyclables, Green = Wet Compost, Red/Special = E-waste/Hazardous).\n"
            f"3. Provide practical tips like rinsing, caps, sorting, or creative DIY upcycling ideas.\n"
            f"4. Keep your tone cheerful and encouraging."
        )

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {"contents": [{"parts": [{"text": prompt}]}]}

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"[EcoAgent Chat] API fallback triggered: {e}")

    # Offline Intelligent Eco Knowledge Engine (NLP rule-based fallback)
    msg = user_msg.lower()

    if "pizza" in msg or "greas" in msg or "oil" in msg:
        return "🍕 Greasy pizza boxes or paper soaked in oil/cheese cannot be recycled with clean paper because grease ruins paper pulp! Tear off the clean dry lid for the Blue Recycling Bin, and place the greasy portion into the Green Compost/Wet Waste Bin."
    elif "batter" in msg or "ewaste" in msg or "e-waste" in msg or "phone" in msg or "electronic" in msg or "cable" in msg:
        return "⚡ E-Waste and batteries contain toxic heavy metals (lithium, lead, mercury) that leach into soil if landfilled. Never place them in ordinary bins; deposit them at specialized municipal e-waste collection bins or authorized electronic drop-offs."
    elif "wash" in msg or "clean" in msg or "rinse" in msg or "cap" in msg or "lid" in msg:
        return "🧴 Yes! For containers (bottles, jars, cans), give them a quick water rinse to remove food/drink residues. Removing food residue prevents contamination. Bottle caps and lids can usually be screwed back on tightly before placing them in the Blue Bin."
    elif "bottle" in msg or "plastic" in msg:
        return "🧴 For plastic bottles: Empty any remaining liquids, give it a quick water rinse, and crush it to save space in the bin. Bottle caps can generally be left on. Always dispose of them in the Blue Recycling Bin!"
    elif "metal" in msg or "can" in msg or "tin" in msg or "aluminum" in msg:
        return "🥫 Metal and aluminum cans are 100% infinitely recyclable without quality loss! Rinse them out, lightly crush to save space, and place them in the Blue Recycling Bin."
    elif "glass" in msg or "jar" in msg:
        return "🍾 Glass bottles and jars are 100% recyclable. Rinse them before recycling. Important: Heat-resistant glassware (Pyrex), mirrors, ceramic mugs, and broken glass should not go in standard glass recycling due to varying melting points—wrap them safely in old paper for general waste."
    elif "paper" in msg or "cardboard" in msg or "box" in msg:
        return "📦 Keep paper and cardboard clean and completely dry. Always flatten cardboard boxes before disposal to preserve bin capacity. Shredded paper or greasy paper goes to compost/wet waste."
    elif "organic" in msg or "food" in msg or "compost" in msg or "peel" in msg or "kitchen" in msg:
        return "🌱 Food scraps, vegetable peels, coffee grounds, and garden clippings belong in the Green Bin for composting. Composting diverts waste from landfills and generates rich organic soil fertilizer while curbing methane emissions!"
    elif "upcycle" in msg or "diy" in msg or "reuse" in msg or "craft" in msg:
        if detected_class == "Plastic":
            return "🎨 Upcycle Idea: Cut plastic bottles in half to create self-watering herb planters, desk pen organizers, or hanging bird feeders!"
        elif detected_class == "Glass":
            return "🎨 Upcycle Idea: Soak off the label and paint the glass jar to make a rustic candle votive, spice jar, or decorative flower vase!"
        elif detected_class == "Metal":
            return "🎨 Upcycle Idea: Wash tin cans and punch pinhole patterns into them with a nail to create glowing lanterns, or wrap with twine for aesthetic desk cups!"
        else:
            return "🎨 Upcycle Idea: Empty containers can be repurposed into drawer organizers, cardboard into storage dividers, and jars into aesthetic home decor before considering disposal!"
    elif "bin" in msg or "color" in msg or "dustbin" in msg:
        return "🗑 Standard Bin Color Guide:\n• 🟢 Green Bin: Organic / Wet Kitchen Waste (Compostable)\n• 🔵 Blue Bin: Dry Recyclables (Plastic, Paper, Glass, Metal)\n• 🔴 Red / Black Bin: Hazardous, E-Waste & Sanitary Waste."
    elif "hello" in msg or "hi" in msg or "hey" in msg:
        return "👋 Hello! I'm EcoAgent, your AI Sustainability Assistant. Ask me anything about waste sorting, recycling guidelines, compost tips, or creative DIY upcycling ideas!"
    else:
        ctx_mention = f"Regarding {detected_class}: " if detected_class and detected_class != "None" else ""
        return f"💡 {ctx_mention}Remember the 3 R's: Reduce, Reuse, Recycle! Always ensure dry recyclables are clean before binning. Feel free to ask me about specific items (like pizza boxes, bottle caps, or e-waste)!"


@app.route("/chat", methods=["POST"])
def chat():
    """
    API endpoint for EcoAgent interactive chat.
    Accepts: { message: str, detected_class: str, confidence: str }
    Returns: { reply: str }
    """
    data = request.get_json(silent=True) or request.form or {}
    user_msg = str(data.get("message", "")).strip()
    detected_class = str(data.get("detected_class", "None"))
    confidence = str(data.get("confidence", "None"))

    if not user_msg:
        return jsonify({"reply": "Please ask a question about waste disposal or recycling!"})

    reply = get_ai_chat_response(user_msg, detected_class, confidence)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
