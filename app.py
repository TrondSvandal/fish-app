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
IMG_SIZE = 384  # CRITICAL: DINOv3 standardizes on 16px patch boundaries

MODEL_PATH = "best_fish_size_estimator_dinov3_vits16.pth"
MODEL_ID_3 = "1c9RHB-6tNY7YMBKg94wWKVLTymzCG7mX"

# -------------------------
# DOWNLOAD MODEL IF NEEDED
# -------------------------
if not os.path.exists(MODEL_PATH):
    print("Downloading DINOv3 model from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_ID_3}"
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
    def __init__(self, target_size=384):
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
# DINOv3 BACKBONE ARCHITECTURE SKELETON
# ==============================================================================
class MetaDinoBlock(nn.Module):
    """Perfect key-matching block utilizing native optimized attention."""
    def __init__(self, dim=384, num_heads=6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ls1 = nn.Parameter(torch.ones(dim)) 
        
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.ls2 = nn.Parameter(torch.ones(dim)) 
        self.num_heads = num_heads

    def forward(self, x):
        B, N, C = x.shape
        normed = self.norm1(x)
        qkv = self.qkv(normed).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        x = x + self.ls1 * self.proj(attn_out)
        
        mlp_out = self.fc2(torch.nn.functional.gelu(self.fc1(self.norm2(x))))
        x = x + self.ls2 * mlp_out
        return x


class DinoSizeEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        # Blueprint structural pipeline mirroring your single-task checkpoint
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 384, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.storage_tokens = nn.Parameter(torch.zeros(1, 4, 384))
        self.blocks = nn.ModuleList([MetaDinoBlock() for _ in range(12)])
        self.norm = nn.LayerNorm(384)
        
        # Sizing Head
        self.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        storage_tokens = self.storage_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, storage_tokens, x), dim=1)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        features = torch.mean(x[:, 5:, :], dim=1)
        return self.head(features)


# Initialize Model layout
model = DinoSizeEstimator()

print(f"Loading local DINOv3 model checkpoint state dict map...")
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()
print("-> DINOv3 Single-Task Sizing Server Engine initialized.")

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

    # Image tensor shape configuration formatting -> [1, 3, 384, 384]
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        size_logits = model(image_tensor)

    # Size Resolution: Revert target from log-space into real centimeters
    pred_length = torch.exp(size_logits).item()

    return jsonify({
        "predicted_length_cm": round(pred_length, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)