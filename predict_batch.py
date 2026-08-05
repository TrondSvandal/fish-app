import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


# ==========================================================
# CONFIG
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 384

MODEL_PATH = "best_fish_size_estimator_dinov3_vits16.pth"

TEST_FOLDER = "test_img"

OUTPUT_CSV = "fish_predictions.csv"


# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

class SquarePadAndResize:

    def __init__(self, target_size=384):
        self.target_size = target_size

    def __call__(self, img):

        w, h = img.size

        max_wh = max(w, h)

        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)

        padding = (hp, vp, hp, vp)

        img = TF.pad(
            img,
            padding,
            fill=0
        )

        img = TF.resize(
            img,
            (self.target_size, self.target_size)
        )

        return img



transform = transforms.Compose([

    SquarePadAndResize(IMG_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])



# ==========================================================
# DINO BLOCK
# ==========================================================

class MetaDinoBlock(nn.Module):

    def __init__(self, dim=384, num_heads=6):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.qkv = nn.Linear(
            dim,
            dim * 3
        )

        self.proj = nn.Linear(
            dim,
            dim
        )

        self.ls1 = nn.Parameter(
            torch.ones(dim)
        )


        self.norm2 = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(
            dim,
            dim * 4
        )

        self.fc2 = nn.Linear(
            dim * 4,
            dim
        )

        self.ls2 = nn.Parameter(
            torch.ones(dim)
        )

        self.num_heads = num_heads



    def forward(self,x):

        B,N,C=x.shape


        qkv=self.qkv(
            self.norm1(x)
        )

        qkv=qkv.reshape(
            B,
            N,
            3,
            self.num_heads,
            C//self.num_heads
        )


        q=qkv[:,:,0].transpose(1,2)
        k=qkv[:,:,1].transpose(1,2)
        v=qkv[:,:,2].transpose(1,2)


        attn=torch.nn.functional.scaled_dot_product_attention(
            q,k,v
        )


        attn=attn.transpose(
            1,
            2
        ).reshape(
            B,N,C
        )


        x=x+self.ls1*self.proj(attn)


        mlp=self.fc2(
            torch.nn.functional.gelu(
                self.fc1(
                    self.norm2(x)
                )
            )
        )


        x=x+self.ls2*mlp


        return x




# ==========================================================
# MODEL
# ==========================================================


class DinoSizeEstimator(nn.Module):

    def __init__(self):

        super().__init__()


        self.patch_embed=nn.Module()

        self.patch_embed.proj=nn.Conv2d(
            3,
            384,
            kernel_size=16,
            stride=16
        )


        self.cls_token=nn.Parameter(
            torch.zeros(1,1,384)
        )


        self.storage_tokens=nn.Parameter(
            torch.zeros(1,4,384)
        )


        self.blocks=nn.ModuleList(
            [
                MetaDinoBlock()
                for _ in range(12)
            ]
        )


        self.norm=nn.LayerNorm(384)



        self.head=nn.Sequential(

            nn.Linear(
                384,
                256
            ),

            nn.LayerNorm(256),

            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                1
            )
        )



    def forward(self,x):

        B=x.shape[0]


        x=self.patch_embed.proj(x)

        x=x.flatten(2).transpose(1,2)


        cls=self.cls_token.expand(
            B,-1,-1
        )


        storage=self.storage_tokens.expand(
            B,-1,-1
        )


        x=torch.cat(
            (
                cls,
                storage,
                x
            ),
            dim=1
        )


        for block in self.blocks:

            x=block(x)



        x=self.norm(x)


        features=torch.mean(
            x[:,5:,:],
            dim=1
        )


        return self.head(features)




# ==========================================================
# LOAD MODEL
# ==========================================================


model=DinoSizeEstimator()


state_dict=torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    state_dict
)


model.to(DEVICE)

model.eval()


print("Model loaded")



# ==========================================================
# DATASET EVALUATION
# ==========================================================


pattern=re.compile(
    r'(\d+(?:\.\d+)?)cm',
    re.IGNORECASE
)


true_lengths=[]
pred_lengths=[]
filenames=[]



extensions=(
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)



with torch.inference_mode():


    for filename in sorted(
        os.listdir(TEST_FOLDER)
    ):


        if not filename.lower().endswith(
            extensions
        ):
            continue



        match=pattern.search(
            filename
        )


        if match is None:

            print(
                "Skipping:",
                filename
            )

            continue



        true=float(
            match.group(1)
        )



        img=Image.open(
            os.path.join(
                TEST_FOLDER,
                filename
            )
        ).convert("RGB")



        tensor=transform(img)

        tensor=tensor.unsqueeze(0).to(
            DEVICE
        )


        prediction=torch.exp(
            model(tensor)
        ).item()



        true_lengths.append(true)

        pred_lengths.append(prediction)

        filenames.append(filename)



# numpy conversion

true_lengths=np.array(
    true_lengths
)

pred_lengths=np.array(
    pred_lengths
)



errors=pred_lengths-true_lengths

absolute=np.abs(errors)



# ==========================================================
# METRICS
# ==========================================================


MAE=absolute.mean()

RMSE=np.sqrt(
    np.mean(errors**2)
)

MAPE=np.mean(
    absolute/true_lengths
)*100


BIAS=np.mean(errors)



print("\n=========================")

print(
    f"Samples : {len(true_lengths)}"
)

print(
    f"MAE     : {MAE:.2f} cm"
)

print(
    f"RMSE    : {RMSE:.2f} cm"
)

print(
    f"MAPE    : {MAPE:.2f}%"
)

print(
    f"BIAS    : {BIAS:+.2f} cm"
)

print("=========================")




# ==========================================================
# SAVE CSV
# ==========================================================


with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as f:


    writer=csv.writer(f)


    writer.writerow(
        [
            "filename",
            "true_cm",
            "prediction_cm",
            "error_cm"
        ]
    )


    for a,b,c,d in zip(
        filenames,
        true_lengths,
        pred_lengths,
        errors
    ):

        writer.writerow(
            [
                a,
                b,
                c,
                d
            ]
        )



print(
    "Saved:",
    OUTPUT_CSV
)




# ==========================================================
# WORST ERRORS
# ==========================================================


print("\nWorst predictions:")


ranking=sorted(
    zip(
        filenames,
        true_lengths,
        pred_lengths,
        errors
    ),
    key=lambda x:abs(x[3]),
    reverse=True
)


for r in ranking[:10]:

    print(
        f"{r[0]:30s} "
        f"True={r[1]:6.1f} "
        f"Pred={r[2]:6.1f} "
        f"Error={r[3]:+6.1f}"
    )




# ==========================================================
# SCATTER PLOT
# ==========================================================


plt.figure(figsize=(7,7))


plt.scatter(
    true_lengths,
    pred_lengths,
    alpha=0.8
)



mn=min(
    true_lengths.min(),
    pred_lengths.min()
)

mx=max(
    true_lengths.max(),
    pred_lengths.max()
)



plt.plot(
    [mn,mx],
    [mn,mx],
    linestyle="--"
)


plt.xlabel(
    "True length (cm)"
)

plt.ylabel(
    "Predicted length (cm)"
)


plt.title(
    f"Prediction vs True\nMAE={MAE:.2f} cm RMSE={RMSE:.2f} cm"
)


plt.grid()

plt.show()



# ==========================================================
# ERROR HISTOGRAM
# ==========================================================


plt.figure(figsize=(7,4))


plt.hist(
    errors,
    bins=20
)


plt.xlabel(
    "Error (cm)"
)

plt.ylabel(
    "Count"
)

plt.title(
    "Prediction Error Distribution"
)


plt.grid()

plt.show()



# ==========================================================
# RESIDUAL PLOT
# ==========================================================


plt.figure(figsize=(7,5))


plt.scatter(
    true_lengths,
    errors
)


plt.axhline(
    0,
    linestyle="--"
)


plt.xlabel(
    "True length (cm)"
)


plt.ylabel(
    "Prediction error (cm)"
)


plt.title(
    "Residuals vs Fish Length"
)


plt.grid()

plt.show()