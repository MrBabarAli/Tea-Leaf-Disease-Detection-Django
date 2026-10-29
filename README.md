# 🍃 Tea Leaf Disease Detection (Django + TensorFlow)

A deep learning web application that detects tea leaf diseases and pest damage from a single uploaded leaf image. A CNN classifier built with TensorFlow/Keras is served through a Django web interface, so users can upload a photo and get the predicted class with a confidence score.

## Overview

Tea is an important cash crop, and diseases and pests such as blight and mites can cause major yield loss. Early identification normally depends on expert inspection. This project provides an automated image-based diagnosis tool that can support farmers and agronomists.

## Detected Classes

The model classifies a leaf image into 7 categories:

| # | Class |
|---|-------|
| 1 | Tea algal leaf spot |
| 2 | Brown Blight |
| 3 | Gray Blight |
| 4 | Helopeltis |
| 5 | Red spider |
| 6 | Green mirid bug |
| 7 | Healthy leaf |

## Features

- Upload a tea leaf image through a web interface
- Real-time prediction with a confidence score
- 7-class classification (diseases, pests, healthy)
- Standalone script (`test_model.py`) to test the model from the command line
- Lightweight Django backend, easy to run locally

## Tech Stack

- **Deep learning:** Python, TensorFlow, Keras
- **Web framework:** Django
- **Image processing:** Pillow, NumPy
- **Database:** SQLite (default Django)

## Dataset

Trained on the **TeaLeafBD** dataset (tea leaf disease and pest images, 7 classes).
- Source: [add dataset link, e.g. Mendeley Data / Kaggle]
- Total images: [add number]
- Train / validation / test split: [e.g. 70 / 15 / 15]
- Input size: 224 × 224 RGB

## Model

- Architecture: [e.g. custom CNN / MobileNetV2 / EfficientNetB0 (transfer learning)]
- Input: 224 × 224 × 3
- Output: 7-class softmax
- Optimizer / loss: [e.g. Adam / categorical cross-entropy]
- Epochs / batch size: [add]
- Trained model file: `tea_leaf_model.keras`

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | [XX.XX%] |
| Precision (macro) | [XX.XX%] |
| Recall (macro) | [XX.XX%] |
| F1-score (macro) | [XX.XX%] |

*(Add the confusion matrix and training curves here, e.g. `![Confusion Matrix](docs/confusion_matrix.png)`)*

## Project Structure

```
Tea-Leaf-Disease-Detection-Django/
├── detector/              # Django app: views, URLs, prediction logic
├── tea_project/           # Django project settings and configuration
├── media/                 # Uploaded images
├── tea_leaf_model.keras   # Trained Keras model
├── test_model.py          # Command-line test script for the model
├── manage.py
└── requirements.txt
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/MrBabarAli/Tea-Leaf-Disease-Detection-Django.git
cd Tea-Leaf-Disease-Detection-Django

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Apply migrations
python manage.py migrate

# 5. Run the server
python manage.py runserver
```

Open **http://127.0.0.1:8000/** in your browser.

## Usage

1. Open the web app.
2. Upload a clear photo of a tea leaf.
3. Click **Predict**.
4. View the predicted class and confidence score.

### Test the model without the web app

Edit `img_path` in `test_model.py` to point to an image, then run:

```bash
python test_model.py
```

Example output:

```
Class: Brown Blight
Confidence: 0.98
```

## Limitations

- Trained on a single dataset, so accuracy may drop on photos taken in very different lighting or backgrounds.
- Intended as a decision-support tool, not a replacement for expert diagnosis.

## Future Work

- Test on field images with varied backgrounds
- Add Grad-CAM visualizations to explain predictions
- Deploy online and build a mobile-friendly version
- Compare against Vision Transformer models

## Author

**Babar Ali**
BS Artificial Intelligence, The Islamia University of Bahawalpur
📧 mralibabar@outlook.com
🔗 GitHub: [MrBabarAli](https://github.com/MrBabarAli)

## License

This project is released under the MIT License.
