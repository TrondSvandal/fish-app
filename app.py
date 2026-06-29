import os
import torch
import torch.nn as nn
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torchvision.transforms as transforms
import gdown

# -------------------------
# CONFIG
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CRITICAL: DINOv2 uses patch sizes of 14, image dimension must be divisible by 14
IMG_SIZE = 392  

MODEL_PATH = "best_dino_fish_size_estimator.pth"
# Updated to use your specified DINOv2 weights file ID
#MODEL_ID_2 = "1zNacRUxXxhyRTnXJpvl6vswz6F6gt3Jv"  
MODEL_ID_3 = "1usVuTs03l-k16bSFqfO40ooiJYY702bx"
# -------------------------
# DOWNLOAD MODEL IF NEEDED
# -------------------------
if not os.path.exists(MODEL_PATH):
    print("Downloading DINOv2 model from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_ID_3}"
    gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

# -------------------------
# DINOv2 MODEL ARCHITECTURE
# -------------------------
class DinoSizeEstimator(nn.Module):
    def __init__(self, model_variant="dinov2_vitb14"):
        super().__init__()
        # Automatically hooks up Meta's baseline DINOv2 framework via PyTorch Hub
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_variant)
        
        if "vits14" in model_variant:
            in_features = 384
        elif "vitb14" in model_variant:
            in_features = 768
        elif "vitl14" in model_variant:
            in_features = 1024
        else:
            raise ValueError("Unknown DINOv2 variant dimensions.")

        # Reconstructed Regression Head layout matching training configuration
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# Initialize wrapper, map weights dictionary, and move to target hardware
model = DinoSizeEstimator(model_variant="dinov2_vitb14")
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

# -------------------------
# TRANSFORM (Adjusted for DINOv2 canvas layout)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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

    # Revert target output values out of log-space back into centimeters
    pred_length = torch.exp(log_pred).item()

    return jsonify({
        "predicted_length_cm": round(pred_length, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)