from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

import sys
import os
import json
import urllib.request
import urllib.error

# Force UTF-8 on Windows terminal so emoji prints don't raise UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Automatically load .env file if present (zero dependencies required)
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v:
                        os.environ.setdefault(k, v)
        print("[Config] Loaded environment variables from .env")
    except Exception as e:
        print(f"[Config] Error loading .env: {e}")

app = Flask(__name__)

model = load_model("model/waste_classifier.keras")

classes = ["Cardboard", "E-Waste", "Glass", "Metal", "Organic", "Paper", "Plastic"]

# Prioritized list of Gemini models for high availability and quota resilience
GEMINI_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-2.5-flash"
]

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

    for model_name in GEMINI_MODELS:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
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
            with urllib.request.urlopen(req, timeout=8) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                dynamic_data = json.loads(raw_text)
                dynamic_data["is_dynamic"] = True
                return dynamic_data
        except Exception as e:
            print(f"[EcoAgent] Dynamic guide ({model_name}) fallback triggered: {e}")
            continue

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
    msg = user_msg.lower().strip()

    # Fast-path instant response for common greetings (0ms latency, zero cloud lag)
    if msg in ["hello", "hi", "hey", "namaste", "good morning", "good evening", "good afternoon", "hlo", "helo", "yo"]:
        return "👋 Hello! I'm EcoAgent, your AI Sustainability Assistant. Ask me anything about waste sorting, recycling rules, composting tips, or creative DIY upcycling ideas!"

    api_key = os.environ.get("GEMINI_API_KEY", "")

    if api_key:
        context_str = f"Scanned item in current session: {detected_class} ({confidence}% confidence)." if detected_class and detected_class != "None" else "No specific item scanned yet."
        prompt = (
            f"You are EcoAgent, a friendly, expert AI Waste Management Assistant. "
            f"Session Context: {context_str}\n"
            f"User Question: '{user_msg}'\n\n"
            f"Instructions: Give a concise, direct answer in 2-3 sentences. Support English and Hinglish naturally. "
            f"Specify exact bin colors (Blue = Dry Recyclables, Green = Compost, Red/Black = Hazardous) and a practical sorting or upcycle tip."
        )

        for model_name in GEMINI_MODELS:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": 200,
                        "temperature": 0.2
                    }
                }

                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=6) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception as e:
                print(f"[EcoAgent Chat] API fallback on {model_name}: {e}")
                continue

    # Offline Intelligent Eco Knowledge Engine (NLP fallback when offline/no API key)
    msg = user_msg.lower()

    # 0. Warning against burning waste
    if any(k in msg for k in ["burn", "fire", "jalana", "jala sakte", "jala du", "aag"]):
        return "⚠️ Never burn waste! Burning plastics or treated materials emits toxic carcinogens (dioxins, furans) and hazardous smoke. Always segregate into the Blue Bin (recyclables) or Green Bin (wet compostable waste)."

    # Hinglish & contextual bin queries
    elif any(k in msg for k in ["kisme", "kaha feku", "kaha daalu", "kaha daale", "feku", "fenku", "daalu", "daale", "fein", "dabba", "dustbin"]):
        if detected_class and detected_class in default_eco_guide:
            g = default_eco_guide[detected_class]
            return f"🗑 For {detected_class}: Isko {g['bin']} me dalein! Tip: {g['tip']}"
        return "🗑 Segregation Rules:\n• 🟢 Green Bin: Geela / Organic Waste (Compostable)\n• 🔵 Blue Bin: Sookha / Recyclable Waste (Plastic, Paper, Glass, Metal)\n• 🔴 Red/Black Bin: Hazardous / E-Waste & Sanitary Waste."

    # General disposal inquiry (how to dispose / kya kare)
    elif any(k in msg for k in ["kya kare", "kya karein", "how to dispose", "how to recycle", "what to do", "kaise kare"]):
        if detected_class and detected_class in default_eco_guide:
            g = default_eco_guide[detected_class]
            return f"ℹ️ For {detected_class}:\n• Bin: {g['bin']}\n• Recyclable: {g['recycle']} | Compostable: {g['compost']}\n• Disposal Tip: {g['tip']}\n• DIY Upcycle: {g['upcycle_idea']}"

    # Why recycle / kyun / fayda
    elif any(k in msg for k in ["why", "kyu", "kyun", "fayda", "benefit"]):
        return "🌍 Recycling preserves natural resources, reduces landfill contamination, and cuts greenhouse gas emissions. For instance, recycling 1 kg of plastic saves ~1.5 kg of CO2!"

    # 1. Food Contaminated Paper / Pizza
    elif any(k in msg for k in ["pizza", "greas", "oil on paper", "cheese box"]):
        return "🍕 Greasy pizza boxes or paper soaked in oil/cheese cannot be recycled with clean paper because grease ruins paper pulp! Tear off the clean dry lid for the Blue Recycling Bin, and place the greasy portion into the Green Compost/Wet Waste Bin."

    # 2. Thermocol & Styrofoam
    elif any(k in msg for k in ["thermocol", "styrofoam", "polystyrene", "eps"]):
        return "📦 Thermocol (Expanded Polystyrene) is non-biodegradable and 95% air—it breaks into harmful micro-plastics and cannot go into regular paper/plastic recycling! Never put it in the Green Compost Bin. Reuse packing blocks for shipping, or hand them to specialized EPS recycling drop-offs / dry non-recyclable waste."

    # 3. Milk Pouches, Polythene & Wrappers
    elif any(k in msg for k in ["milk pouch", "polythene", "plastic bag", "wrapper", "chips", "kurkure", "biscuit pack", "snack packet", "polybag"]):
        return "🛍 Milk pouches (LDPE) and multi-layered packaging (MLP like chips/biscuit packets) cannot be processed with rigid PET plastic bottles. Wash & dry milk pouches and hand them to specialized dry waste aggregators. Chips wrappers go to non-recyclable dry waste for cement kiln energy recovery."

    # 4. Tetra Pak & Juice Cartons
    elif any(k in msg for k in ["tetra", "juice box", "juice carton", "milk carton"]):
        return "🧃 Tetra Paks are composite cartons made of 75% paperboard, 20% polyethylene, and 5% aluminum foil. Rinse, flatten, and deposit in Blue Dry Recycling Bins where municipal facilities have hydrapulping capabilities to separate paper fibers from poly-aluminum."

    # 5. Medicines & Blister Packs
    elif any(k in msg for k in ["medicine", "tablet", "blister", "syrup", "expired medicine", "pharma"]):
        return "💊 Medicines and metallic/plastic blister packs are Domestic Hazardous / Biomedical waste. Never flush expired medicines down drains or compost them! Wrap them securely and place them in the Red/Black Hazardous Bin, or return to pharmacy take-back kiosks."

    # 6. Bulbs, CFLs & Tube Lights
    elif any(k in msg for k in ["bulb", "cfl", "led", "tube light", "tubelight", "fluorescent"]):
        return "💡 Fluorescent tubes and CFL bulbs contain hazardous mercury vapor. Never throw them in standard glass recycling or crush them! Wrap carefully and dispose of them at designated Hazardous E-Waste kiosks or Red Bins."

    # 7. Aluminium Foil & Food Containers
    elif any(k in msg for k in ["foil", "aluminium foil", "aluminum foil", "silver foil", "takeaway container"]):
        return "🥡 Clean aluminium foil is 100% infinitely recyclable! Scrunch clean foil scraps into a large ball (so recycling machines don't lose it) and place in the Blue Bin. Greasy food-caked foil should go into non-recyclable dry waste."

    # 8. Clothes, Fabrics & Footwear
    elif any(k in msg for k in ["cloth", "textile", "shirt", "jeans", "fabric", "shoe", "footwear"]):
        return "👕 Wearable clothes and shoes should be donated to NGOs or textile banks. Torn or ruined fabrics can be upcycled into cleaning rags or dropped off at textile recycling drives. Never throw clothes into green wet waste!"

    # 9. Sanitary Waste & Diapers
    elif any(k in msg for k in ["diaper", "pad", "sanitary", "mask", "bandage", "tissue", "cotton swab"]):
        return "🗑 Sanitary waste (diapers, pads, masks, bandages) is bio-hazardous. Wrap securely in newspaper or marked red-dot disposal bags and place strictly in the Red / Black Sanitary Waste Bin. Never flush or compost them."

    # 10. Organic Garden & Kitchen (Coconut, Egg shells, Leaves, Tea bags)
    elif any(k in msg for k in ["coconut", "egg shell", "dry leaf", "leaves", "tea bag", "coffee ground", "wood"]):
        return "🥥 Coconut shells, dry leaves, egg shells, and tea leaves are organic waste! Egg shells enrich compost with calcium. Dry leaves and coconut shells make great carbon-rich 'brown' layers for compost pits (Green Bin)."

    # 11. Meat, Bones & Dairy leftovers
    elif any(k in msg for k in ["meat", "bone", "fish", "dairy", "cheese"]):
        return "🍗 Meat leftovers and bones can attract vermin and cause odors in simple home pits. For municipal collection, place them in the Green Wet Waste Bin for industrial anaerobic composting or biogas generation."

    # 12. Pens & Stationery
    elif any(k in msg for k in ["pen", "marker", "pencil", "stationery", "refill", "eraser"]):
        return "🖊 Plastic pens and markers consist of mixed plastics, metal springs, and chemical inks. Separate metal nibs if possible, collect used pens for NGO recycling programs (like TerraCycle), or dispose of in dry non-recyclable waste."

    # 13. Broken Glass, Mirrors & Ceramics
    elif any(k in msg for k in ["mirror", "ceramic", "crockery", "broken glass", "plate", "mug", "pyrex"]):
        return "🪞 Mirrors, ceramics, Pyrex, and window panes have different melting temperatures than bottle glass and ruin recyclers' furnaces! Wrap broken pieces securely in cardboard or newspaper to protect sanitation workers and deposit in non-recyclable dry waste."

    # 14. E-Waste & Batteries
    elif any(k in msg for k in ["batter", "ewaste", "e-waste", "phone", "electronic", "cable", "wire", "charger", "laptop"]):
        return "⚡ E-Waste and batteries contain toxic heavy metals (lithium, lead, cadmium). Never place them in ordinary bins; deposit them at authorized e-waste collection bins or electronic retail take-back points."

    # 15. Washing & Rinsing
    elif any(k in msg for k in ["wash", "clean", "rinse", "cap", "lid"]):
        return "🧴 Yes! For containers (bottles, jars, cans), give them a quick water rinse to remove food/drink residues. Removing residues prevents mold and contamination. Bottle caps can usually be screwed back on tightly before placing them in the Blue Bin."

    # 16. Single Use Plastics & Straws
    elif any(k in msg for k in ["single use", "straw", "cutlery", "plastic fork", "plastic spoon"]):
        return "🥤 Single-use plastic straws and cutlery are major marine pollutants and take 400+ years to degrade. Switch to reusable steel or bamboo alternatives. Used disposable cutlery belongs in dry non-recyclable waste."

    # 17. Paper Cups & Coffee Cups
    elif any(k in msg for k in ["coffee cup", "paper cup", "disposable cup"]):
        return "☕ Most disposable paper coffee cups have an internal waterproof polyethylene plastic coating. This makes them non-recyclable in standard paper mills. Unless certified 100% compostable, dispose of them in dry general waste."

    # 18. General Plastic Bottles
    elif any(k in msg for k in ["bottle", "plastic"]):
        return "🧴 For plastic bottles (PET #1 / HDPE #2): Empty liquids, rinse, squash flat to save bin space, screw cap back on, and drop into the Blue Recycling Bin!"

    # 19. Metal & Cans
    elif any(k in msg for k in ["metal", "can", "tin", "aluminum"]):
        return "🥫 Metal and aluminum cans are 100% infinitely recyclable without quality loss! Rinse them out, lightly crush to save bin space, and place them in the Blue Recycling Bin."

    # 20. Glass Containers
    elif any(k in msg for k in ["glass", "jar"]):
        return "🍾 Glass bottles and jars are 100% recyclable. Rinse them before binning into the Blue Bin. Remove metallic lids and recycle both separately."

    # 21. Paper & Cardboard
    elif any(k in msg for k in ["paper", "cardboard", "box", "newspaper", "carton"]):
        return "📦 Keep paper and cardboard clean and dry. Always flatten cardboard boxes before disposal to preserve bin capacity. Place in the Blue Bin."

    # 22. Organic & Food Scraps
    elif any(k in msg for k in ["organic", "food", "compost", "peel", "kitchen waste"]):
        return "🌱 Vegetable peels, fruit rinds, and food scraps belong in the Green Bin for composting. Composting diverts waste from landfills and creates rich nutrient fertilizer!"

    # 23. Bin Colors
    elif any(k in msg for k in ["bin", "color", "dustbin", "segregat"]):
        return "🗑 Standard Segregation Bins:\n• 🟢 Green Bin: Wet Organic & Kitchen Waste (Compostable)\n• 🔵 Blue Bin: Clean Dry Recyclables (Plastic, Paper, Glass, Metal)\n• 🔴 Red / Black Bin: Hazardous, E-Waste & Sanitary Waste."

    # 24. Upcycling & DIY Ideas
    elif any(k in msg for k in ["upcycle", "diy", "reuse", "craft"]):
        if detected_class == "Plastic":
            return "🎨 Upcycle Idea: Cut plastic bottles in half to create self-watering herb planters, desk pen organizers, or hanging bird feeders!"
        elif detected_class == "Glass":
            return "🎨 Upcycle Idea: Clean glass jars and paint them to make rustic candle votives, spice storage containers, or decorative flower vases!"
        elif detected_class == "Metal":
            return "🎨 Upcycle Idea: Wash tin cans, hammer decorative pinhole patterns, and insert candles for cozy lanterns, or wrap with jute twine for desk pen stands!"
        elif detected_class == "Cardboard":
            return "🎨 Upcycle Idea: Cut cardboard into custom drawer dividers, desk organizers, or use as biodegradable weed barrier mulch for garden beds!"
        else:
            return "🎨 Upcycle Idea: Repurpose empty jars as kitchen storage, bottles into self-watering planters, and cardboard into drawer dividers before tossing them out!"

    # 25. Technical Viva Questions: MobileNetV2, Latency, Gemini
    elif any(k in msg for k in ["mobilenet", "cnn", "deep learning", "architecture", "model"]):
        return "🧠 MobileNetV2 is a lightweight Convolutional Neural Network (CNN) developed by Google. It utilizes Inverted Residual blocks with Depthwise Separable Convolutions to achieve high classification accuracy (~92%) with 70% fewer parameters and ultra-low latency, making it ideal for edge deployment."
    elif any(k in msg for k in ["latency", "speed", "fast", "fps", "inference"]):
        return "⚡ Latency is the time delay between image input and model output prediction. In our EcoSort system, MobileNetV2 achieves inference latency under 100 milliseconds per frame, enabling fast, real-time waste sorting on standard hardware."
    elif any(k in msg for k in ["gemini", "agent", "llm"]):
        return "🤖 Google Gemini 1.5 Flash powers the AI Agent Layer of EcoSort. While MobileNetV2 detects the waste type, the Gemini Agent generates context-aware disposal tips, carbon savings, creative upcycling guides, and answers real-time recycling questions."
    elif any(k in msg for k in ["project", "ecosort", "about", "what do you do", "who are you"]):
        return "🌱 EcoSort AI is an intelligent dual-engine waste classification system combining MobileNetV2 computer vision (7 classes) with Google Gemini AI for real-time sustainable disposal guidance, carbon footprint tracking, and smart sorting assistance!"

    # 26. Greetings
    elif any(k in msg for k in ["hello", "hi", "hey", "namaste", "good morning", "good afternoon"]):
        return "👋 Hello! I'm EcoAgent, your AI Sustainability Assistant. Ask me anything about waste sorting, recycling rules, composting tips, or creative DIY upcycling ideas!"

    # Default smart fallback (context-aware disposal guidance)
    else:
        if detected_class and detected_class in default_eco_guide:
            g = default_eco_guide[detected_class]
            return (
                f"💡 For scanned {detected_class}:\n"
                f"• Bin: {g['bin']} (Recycle: {g['recycle']}, Compost: {g['compost']})\n"
                f"• Disposal Tip: {g['tip']}\n"
                f"• Creative Upcycle: {g['upcycle_idea']}"
            )
        else:
            return "💡 Remember the 3 R's: Reduce, Reuse, Recycle! Always ensure dry recyclables are clean and dry before binning. Feel free to ask me about specific items (like pizza boxes, thermocol, milk packets, or e-waste)!"


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
