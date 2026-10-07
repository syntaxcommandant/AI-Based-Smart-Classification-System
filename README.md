# EcoSort AI: Smart Waste Classification & Sustainability Agent System

## Overview
EcoSort AI is an intelligent waste classification and sustainability recommendation system. It leverages Deep Learning (MobileNetV2 Transfer Learning) to classify waste into 7 distinct categories and employs an **AI Agent layer** to generate dynamic disposal guidelines, carbon footprint savings, and creative DIY upcycling ideas.

---

## 7 Waste Categories
1. **Cardboard** (Recyclable paperboard)
2. **E-Waste** (Electronic components, hazardous)
3. **Glass** (Recyclable glass containers/bottles)
4. **Metal** (Tin cans, beverage cans, scraps)
5. **Organic** (Food/kitchen waste, biodegradable)
6. **Paper** (Clean paper, newspapers, magazines)
7. **Plastic** (PET bottles, containers, recyclable polymers)

---

## Key Features
- **Deep Learning Classification:** MobileNetV2 fine-tuned with data augmentation for high-accuracy multi-class inference.
- **AI Agent Sustainability Guide:** Real-time actionable guidance providing:
  - Designated Colored Disposal Bin
  - Compostability & Recyclability status
  - Item-specific disposal tips
  - Estimated Carbon Footprint (CO₂ saved)
  - Creative DIY Upcycling & Repurposing ideas
- **Fail-Safe Offline Mode:** Graceful fallback ensures 100% system availability even without an internet connection or API keys.
- **Modern Glassmorphic Web Interface:** Built with responsive vanilla CSS and AJAX image scanning preview.

---

## Technologies Used
- **Backend:** Python, Flask
- **Computer Vision & Deep Learning:** TensorFlow / Keras (MobileNetV2)
- **Evaluation & Metrics:** Scikit-Learn, Seaborn, Matplotlib
- **Frontend:** HTML5, CSS3 (Custom Glassmorphic Theme), JavaScript (Fetch API)

---

## Project Structure
```text
Mini Project/
│── app.py                 # Flask server with AI Agent layer
│── train.py               # MobileNetV2 transfer learning training pipeline
│── evaluate.py            # Confusion matrix & classification report generation
│── predict.py             # Single image standalone inference script
│── requirements.txt       # Project dependencies
│── README.md              # Project documentation
│── templates/
│     └── index.html       # Web application UI
│── uploads/               # Temporary storage for uploaded images
│── model/
│     └── waste_classifier.keras  # Trained deep learning model
```

---

## How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Set Gemini API Key for Dynamic AI Agent
```bash
# On Windows PowerShell:
$env:GEMINI_API_KEY="your_api_key_here"

# On Windows CMD:
set GEMINI_API_KEY=your_api_key_here
```
*(If no API key is provided, the system seamlessly operates in offline mode with enriched default sustainability guidelines.)*

### 3. Start the Application
```bash
python app.py
```

### 4. Open in Browser
Navigate to:
```
http://127.0.0.1:5000
```

---

## Model Evaluation
To generate the Confusion Matrix and Classification Report:
```bash
python evaluate.py
```
