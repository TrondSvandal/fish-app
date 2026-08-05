import os
import torch
import torch.nn as nn
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import gdown

# System Tuning
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True

# -------------------------
# CONFIG
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CRITICAL: DINOv2 uses patch sizes of 14 (392 / 14 = 28 patches)
IMG_SIZE = 392  

MODEL_PATH = "best_fish_size_estimator_dinov2_vits14_V4.pth"
MODEL_ID_DINOV2 = "1nQmM127URLmMi0CvHMVpLYdJlFXDdPis"

# -------------------------
# DOWNLOAD MODEL IF NEEDED
# -------------------------
if not os.path.exists(MODEL_PATH):
    print("Downloading DINOv2 model from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_ID_DINOV2}"
    gdown.download(url, MODEL_PATH, quiet=False)


# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

# ==============================================================================
# SQUISH-FREE TRANSFORMS
# ==============================================================================
class SquarePadAndResize(object):
    """Pads image evenly to a square maintaining true native aspect proportions."""
    def __init__(self, target_size=392):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        
        img = TF.pad(img, padding, fill=0, padding_mode='constant')
        img = TF.resize(img, (self.target_size, self.target_size))
        return img

# ==============================================================================
# DINOv2 BACKBONE ARCHITECTURE
# ==============================================================================
class DinoSizeEstimator(nn.Module):
    def __init__(self, model_variant="dinov2_vits14"):
        super().__init__()
        # Load official DINOv2 ViT-Small backbone
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 
            model_variant, 
            pretrained=False
        )
        
        # Sizing Regression Head
        self.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# Initialize Model layout
model = DinoSizeEstimator(model_variant="dinov2_vits14")

print("Loading DINOv2 model checkpoint state dict map...")
raw_state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# Clean '_orig_mod.' prefix if checkpoint was saved with torch.compile
clean_state_dict = {}
for k, v in raw_state_dict.items():
    new_key = k.replace("_orig_mod.", "")
    clean_state_dict[new_key] = v

model.load_state_dict(clean_state_dict)

model.to(DEVICE)
model.eval()
print("-> DINOv2 Sizing Server Engine initialized successfully.")

# -------------------------
# SQUISH-FREE PIPELINE TRANSFORM
# -------------------------
transform = transforms.Compose([
    SquarePadAndResize(target_size=IMG_SIZE), # Preserves physical shapes
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

    # Image tensor shape configuration formatting -> [1, 3, 392, 392]
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        log_pred = model(image_tensor)

    # Size Resolution: Revert target from log-space into real centimeters
    pred_length = torch.exp(log_pred).item()

    return jsonify({
        "predicted_length_cm": round(pred_length, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)