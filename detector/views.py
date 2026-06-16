from django.shortcuts import render
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load Model
model = tf.keras.models.load_model(
    "tea_leaf_model.keras",
    compile=False
)

# Class Names
class_names = [
    "Tea algal leaf spot",
    "Brown Blight",
    "Gray Blight",
    "Helopeltis",
    "Red spider",
    "Green mirid bug",
    "Healthy leaf"
]

# Disease Information
disease_info = {

    "Tea algal leaf spot": {
        "symptoms": "Orange-red circular spots appear on leaves.",
        "treatment": "Apply copper-based fungicides and remove infected leaves.",
        "prevention": "Maintain proper plant spacing and reduce humidity."
    },

    "Brown Blight": {
        "symptoms": "Brown lesions spread across the leaf surface.",
        "treatment": "Use recommended fungicides and prune infected leaves.",
        "prevention": "Avoid excess moisture and improve airflow."
    },

    "Gray Blight": {
        "symptoms": "Gray necrotic spots with dark borders on leaves.",
        "treatment": "Apply fungicides and remove infected plant material.",
        "prevention": "Monitor plantations regularly and maintain field hygiene."
    },

    "Helopeltis": {
        "symptoms": "Brown feeding marks and damaged young shoots.",
        "treatment": "Apply approved insecticides against Helopeltis.",
        "prevention": "Regular pest monitoring and early control."
    },

    "Red spider": {
        "symptoms": "Leaves become bronze, dry, and curl.",
        "treatment": "Apply miticides and maintain adequate irrigation.",
        "prevention": "Inspect plants regularly and maintain humidity."
    },

    "Green mirid bug": {
        "symptoms": "Damage to young leaves and tender shoots.",
        "treatment": "Use recommended insecticides and remove affected shoots.",
        "prevention": "Regular field scouting and pest management."
    },

    "Healthy leaf": {
        "symptoms": "No disease detected.",
        "treatment": "No treatment required.",
        "prevention": "Continue good agricultural practices."
    }
}


def predict_image(img_path):

    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    print("Prediction Vector:", prediction)

    index = np.argmax(prediction)

    disease = class_names[index]
    confidence = float(np.max(prediction))

    # Severity
    if confidence >= 0.90:
        severity = "High Risk 🔴"
    elif confidence >= 0.70:
        severity = "Moderate Risk 🟠"
    else:
        severity = "Low Risk 🟢"

    return (
        disease,
        confidence,
        severity,
        disease_info[disease]
    )


def home(request):

    result = None
    confidence = None
    severity = None
    info = None
    image_url = None

    if request.method == "POST":

        image = request.FILES["image"]

        # Create media folder automatically
        os.makedirs("media", exist_ok=True)

        # Save uploaded image
        img_path = os.path.join("media", image.name)

        with open(img_path, "wb+") as f:
            for chunk in image.chunks():
                f.write(chunk)

        # Image URL for HTML
        image_url = "/" + img_path.replace("\\", "/")

        # Prediction
        result, confidence, severity, info = predict_image(img_path)

    return render(
        request,
        "index.html",
        {
            "result": result,
            "confidence": confidence,
            "severity": severity,
            "info": info,
            "image_url": image_url
        }
    )