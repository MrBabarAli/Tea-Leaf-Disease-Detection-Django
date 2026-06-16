import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model(
    "tea_leaf_model.keras",
    compile=False
)

class_names = [
    "Tea algal leaf spot",
    "Brown Blight",
    "Gray Blight",
    "Helopeltis",
    "Red spider",
    "Green mirid bug",
    "Healthy leaf"
]

img_path = r"C:\Users\Hafiz Computer\Downloads\archive (2)\teaLeafBD\teaLeafBD\2. Brown Blight\brown_blight_00450.jpg"

img = Image.open(img_path).convert("RGB")
img = img.resize((224, 224))

img = np.array(img)
img = np.expand_dims(img, axis=0)

pred = model.predict(img)

print("Prediction Vector:")
print(pred)

print("Class:", class_names[np.argmax(pred)])
print("Confidence:", np.max(pred))