# AI-Based Smart Waste Classification System

## Overview
This project is an AI-based waste classification system that classifies waste into two categories:
- Wet Waste
- Dry Waste

The model is built using Convolutional Neural Network (CNN) and deployed using Flask.

## Features
- Upload an image
- Predict whether waste is Wet or Dry
- User-friendly web interface
- Deep Learning based classification

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Flask
- HTML

## Project Structure

Mini Project/
│── app.py
│── train.py
│── predict.py
│── requirements.txt
│── README.md
│── templates/
│     └── index.html
│── uploads/
│── dataset/ (ignored)
│── model/ (ignored)

## Dataset
Dataset was collected from Kaggle.
- Organic Waste → Wet Waste
- Recyclable + Non-Recyclable → Dry Waste

## How to Run

Install dependencies

pip install -r requirements.txt

Run

python app.py

Open

http://127.0.0.1:5000

## Future Scope
- Multi-class waste classification
- Real-time camera detection
- Mobile application
- Smart dustbin integration

