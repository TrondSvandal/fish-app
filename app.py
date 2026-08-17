import os
import torch
import torch.nn as nn

from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageFile

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import gdown


# ==============================================================================
# SYSTEM TUNING
# ==============================================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True


# ==============================================================================
# CONFIG
# ==============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# DINOv2 ViT-S/14 uses 14x14 patches.
# 392 / 14 = 28 patches per side.
IMG_SIZE = 518


# ==============================================================================
# MODEL
# ==============================================================================

MODEL_PATH = "best_fish_size_estimator_dinov2_vitb14_V6.pth"
MODEL_ID_DINOV2 = "1VkxJXaKHqdTXIhkp4gqS54TcQREgXu2j"


# ==============================================================================
# DOWNLOAD MODEL IF NEEDED
# ==============================================================================

if not os.path.exists(MODEL_PATH):
    print("Downloading trained model from Google Drive...")

    url = f"https://drive.google.com/uc?id={MODEL_ID_DINOV2}"

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False
    )

    print("Model download complete.")


# ==============================================================================
# FLASK APP
# ==============================================================================

app = Flask(__name__)


# ==============================================================================
# SQUISH-FREE TRANSFORMS
# ==============================================================================

class SquarePadAndResize:
    """
    Pads image evenly to a square while maintaining
    the original aspect ratio, then resizes to IMG_SIZE.
    """

    def __init__(self, target_size=392):
        self.target_size = target_size

    def __call__(self, img):

        w, h = img.size

        max_wh = max(w, h)

        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)

        padding = (hp, vp, hp, vp)

        # Pad without distorting the original image
        img = TF.pad(
            img,
            padding,
            fill=0,
            padding_mode="constant"
        )

        # Resize to the model's expected input size
        img = TF.resize(
            img,
            (self.target_size, self.target_size)
        )

        return img


# ==============================================================================
# DINOv2 BACKBONE ARCHITECTURE
# ==============================================================================

class DinoSizeEstimator(nn.Module):

    def __init__(self, model_variant="dinov2_vitb14"):
        super().__init__()

        # DINOv2 ViT-B/14
        # Output feature dimension = 768
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            model_variant,
            pretrained=False
        )

        # EXACT regression head contained in the checkpoint:
        # 768 -> 384 -> 1
        self.head = nn.Sequential(
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ==============================================================================
# INITIALIZE MODEL
# ==============================================================================

print("Initializing DINOv2 ViT-S/14 model...")

model = DinoSizeEstimator(
    model_variant="dinov2_vitb14"
)


# ==============================================================================
# LOAD TRAINED CHECKPOINT
# ==============================================================================

print("Loading trained model checkpoint...")

raw_state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ==============================================================================
# CLEAN CHECKPOINT KEYS
#
# If the model was saved using torch.compile(), keys can contain:
#
#     _orig_mod.
#
# Remove that prefix so they match the normal model architecture.
# ==============================================================================

clean_state_dict = {}

for key, value in raw_state_dict.items():

    new_key = key.replace("_orig_mod.", "")

    clean_state_dict[new_key] = value


# Load trained weights
model.load_state_dict(clean_state_dict)


# ==============================================================================
# MOVE MODEL TO DEVICE
# ==============================================================================

model.to(DEVICE)

model.eval()


print(
    f"-> DINOv2 ViT-S/14 sizing engine initialized successfully on {DEVICE}"
)


# ==============================================================================
# INFERENCE TRANSFORM
# ==============================================================================

transform = transforms.Compose([

    # Preserve original image proportions
    SquarePadAndResize(
        target_size=IMG_SIZE
    ),

    # Convert PIL image to tensor
    transforms.ToTensor(),

    # Same ImageNet normalization used by DINOv2
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================================================================
# ROUTES
# ==============================================================================

@app.route("/")
def home():

    return render_template("index.html")


# ==============================================================================
# PREDICTION ROUTE
# ==============================================================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------------------------
    # Check upload
    # --------------------------------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400


    try:

        # ----------------------------------------------------------------------
        # Load image
        # ----------------------------------------------------------------------

        file = request.files["image"]

        image = Image.open(
            file.stream
        ).convert("RGB")


        # ----------------------------------------------------------------------
        # Preprocess
        #
        # Result before batch dimension:
        #
        # [3, 392, 392]
        # ----------------------------------------------------------------------

        image_tensor = transform(image)


        # Add batch dimension:
        #
        # [1, 3, 392, 392]

        image_tensor = image_tensor.unsqueeze(0)


        # Move to GPU/CPU

        image_tensor = image_tensor.to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------------------------
        # MODEL INFERENCE
        # ----------------------------------------------------------------------

        with torch.inference_mode():

            log_pred = model(image_tensor)


        # ----------------------------------------------------------------------
        # CONVERT LOG PREDICTION BACK TO CENTIMETERS
        #
        # The model predicts log(length).
        #
        # Therefore:
        #
        # length_cm = exp(log_prediction)
        # ----------------------------------------------------------------------

        pred_length_cm = torch.exp(
            log_pred
        ).item()


        # ----------------------------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------------------------

        return jsonify({
            "predicted_length_cm": round(
                pred_length_cm,
                2
            )
        })


    except Exception as e:

        print(
            f"Prediction error: {e}"
        )

        return jsonify({
            "error": str(e)
        }), 500


# ==============================================================================
# START FLASK SERVER
# ==============================================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )

