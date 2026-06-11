import os
import torch
import torch.nn as nn
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torchvision.transforms as transforms
from timm import create_model
import gdown

# -------------------------
# CONFIG
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384
NUM_CLASSES = 205

MODEL_PATH = "best_fish_size_estimator.pth"
MODEL_ID = "1OMcK_K-rgK1-KzNDVx_Qu5S4_aMhk64W"  # Google Drive file ID

# -------------------------
# DOWNLOAD MODEL IF NEEDED
# -------------------------
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

# -------------------------
# MODEL
# -------------------------
def build_size_estimator(num_species_classes=205):
    model = create_model(
        "convnext_small",
        pretrained=False,
        num_classes=num_species_classes
    )

    in_features = model.head.fc.in_features

    model.head.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )

    return model

model = build_size_estimator(NUM_CLASSES)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        log_pred = model(image)

    pred_length = torch.exp(log_pred).item()

    return jsonify({
        "predicted_length_cm": round(pred_length, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)