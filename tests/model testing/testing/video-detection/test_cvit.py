"""Benchmarks CViT2, the Convolutional Vision Transformer of Wodajo and Atnafu
(arXiv:2102.11126).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel, extract_frames, WEIGHTS_DIR


MODEL_KEY = "cvit"
HF_REPO_ID = "Deressa/cvit"
HF_FILENAME = "cvit2_deepfake_detection_ep_50.pth"


# -------------------------------------------------------------------------- CViT2
# architecture, vendored from erprogs/CViT (model/cvit.py).

def _build_cvit_class():
    import torch
    from torch import nn
    from einops import rearrange

    class Residual(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x, **kwargs):
            return self.fn(x, **kwargs) + x

    class PreNorm(nn.Module):
        def __init__(self, dim, fn):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fn = fn

        def forward(self, x, **kwargs):
            return self.fn(self.norm(x), **kwargs)

    class FeedForward(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
            )

        def forward(self, x):
            return self.net(x)

    class Attention(nn.Module):
        def __init__(self, dim, heads=8):
            super().__init__()
            self.heads = heads
            self.scale = dim ** -0.5
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
            self.to_out = nn.Linear(dim, dim)

        def forward(self, x, mask=None):
            b, n, _, h = *x.shape, self.heads
            qkv = self.to_qkv(x)
            q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
            dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
            attn = dots.softmax(dim=-1)
            out = torch.einsum('bhij,bhjd->bhid', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)

    class Transformer(nn.Module):
        def __init__(self, dim, depth, heads, mlp_dim):
            super().__init__()
            self.layers = nn.ModuleList([])
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, heads=heads))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim))),
                ]))

        def forward(self, x, mask=None):
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
            return x

    class CViT(nn.Module):
        def __init__(self, image_size=224, patch_size=7, num_classes=2, channels=512,
                     dim=1024, depth=6, heads=8, mlp_dim=2048):
            super().__init__()
            assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'

            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),

                nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),

                nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),

                nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2),
            )

            num_patches = (image_size // patch_size) ** 2
            self.max_sequence_length = num_patches + 1
            patch_dim = channels * patch_size ** 2
            self.patch_size = patch_size

            self.pos_embedding = nn.Parameter(torch.randn(1, self.max_sequence_length, dim))
            self.patch_to_embedding = nn.Linear(patch_dim, dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.transformer = Transformer(dim, depth, heads, mlp_dim)
            self.to_cls_token = nn.Identity()
            self.mlp_head = nn.Sequential(
                nn.Linear(dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, num_classes)
            )

        def forward(self, img, mask=None):
            p = self.patch_size
            x = self.features(img)
            y = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
            y = self.patch_to_embedding(y)
            cls_tokens = self.cls_token.expand(y.shape[0], -1, -1)
            x = torch.cat((cls_tokens, y), dim=1)
            x += self.pos_embedding[:, :x.size(1)]
            x = self.transformer(x, mask)
            x = self.to_cls_token(x[:, 0])
            return self.mlp_head(x)

    return CViT


class CViTModel(BaseModel):
    name = "CViT2 (Wodajo & Atnafu, CNN+ViT)"

    def __init__(self, device):
        self.device = device
        self.model = None
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self._face_cascade = None
        self.faces_found = 0
        self.faces_missed = 0

    def load(self):
        import torch
        import cv2

        CViT = _build_cvit_class()
        self.torch = torch
        self.model = CViT(image_size=224, patch_size=7, num_classes=2, channels=512,
                           dim=1024, depth=6, heads=8, mlp_dim=2048)

        weights_dir = os.path.join(WEIGHTS_DIR, 'cvit')
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, HF_FILENAME)
        if not os.path.exists(weights_path):
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(
                repo_id=HF_REPO_ID, filename=HF_FILENAME, repo_type="dataset",
                local_dir=weights_dir,
            )

        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

    def _crop_face(self, frame_rgb):
        import cv2
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            self.faces_missed += 1
            return frame_rgb

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        padding = int(0.25 * max(w, h))
        top = max(0, y - padding)
        left = max(0, x - padding)
        bottom = min(frame_rgb.shape[0], y + h + padding)
        right = min(frame_rgb.shape[1], x + w + padding)
        self.faces_found += 1
        return frame_rgb[top:bottom, left:right]

    def _preprocess_frame(self, frame_rgb):
        from PIL import Image
        face = self._crop_face(frame_rgb)
        img = Image.fromarray(face).resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2, 0, 1)
        return self.torch.tensor(arr, dtype=self.torch.float32)

    def score_video(self, video_path, num_frames=20):
        torch = self.torch
        frames = extract_frames(video_path, num_frames=num_frames)
        if not frames:
            raise ValueError("No frames extracted")

        batch = torch.stack([self._preprocess_frame(f) for f in frames]).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)  # [num_frames, 2]
            probs = torch.sigmoid(logits).cpu().numpy()

        # Official repo's real_or_fake(): argmax index 0 = FAKE, 1 = REAL (same convention as
        # Naman712's own comment).
        return float(np.mean(probs[:, 0]))


def main():
    parser = common.build_arg_parser(
        "Evaluate CViT2 (Wodajo & Atnafu's Convolutional Vision Transformer) on a labelled video dataset."
    )
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, CViTModel, args)


if __name__ == "__main__":
    main()
