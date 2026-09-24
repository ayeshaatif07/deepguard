"""Signal 1 - video deepfake detection using CViT2 (Wodajo & Atnafu, arXiv:2102.11126).
Samples frames, face-crops each one, and averages per-frame scores into one verdict.
"""

import os

import numpy as np
import cv2
import torch
from torch import nn
from PIL import Image
from einops import rearrange

HF_REPO_ID = "Deressa/cvit"
HF_FILENAME = "cvit2_deepfake_detection_ep_50.pth"
NUM_FRAMES = 15  # native frame count CViT2 was benchmarked and selected with
IMAGE_SIZE = 224
PATCH_SIZE = 7


# CViT2 architecture, vendored from erprogs/CViT model/cvit.py.

class _Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class _PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class _FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class _Attention(nn.Module):
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


class _Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                _Residual(_PreNorm(dim, _Attention(dim, heads=heads))),
                _Residual(_PreNorm(dim, _FeedForward(dim, mlp_dim))),
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
        self.transformer = _Transformer(dim, depth, heads, mlp_dim)
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


# Production scorer: score_video_frames() and get_verdict().

class CViTScorer:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
        else:
            if device == "mps" and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif device == 0 and torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')

        print(f"Loading CViT2 (Wodajo & Atnafu) on device {self.device}...")

        self.model = CViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=2,
                           channels=512, dim=1024, depth=6, heads=8, mlp_dim=2048)

        weights_dir = os.path.join(os.path.dirname(__file__), 'cvit', 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, HF_FILENAME)
        if not os.path.exists(weights_path):
            print(f"CViT2 weights not found locally, downloading from {HF_REPO_ID}...")
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

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Face-crop before resizing; the model was trained on face-cropped frames.
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

    def _crop_face(self, frame_rgb):
        """Detect the largest face and crop to it with padding. Falls back
        to the full frame if no face is detected."""
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return frame_rgb

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        padding = int(0.25 * max(w, h))
        top = max(0, y - padding)
        left = max(0, x - padding)
        bottom = min(frame_rgb.shape[0], y + h + padding)
        right = min(frame_rgb.shape[1], x + w + padding)
        return frame_rgb[top:bottom, left:right]

    def _preprocess_frame(self, frame_rgb):
        """Crop to face, resize + normalize a single RGB frame (numpy array)."""
        face = self._crop_face(frame_rgb)
        img = Image.fromarray(face).resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2, 0, 1)
        return torch.tensor(arr, dtype=torch.float32)

    def score_video_frames(self, frames):
        if not frames:
            return 0.0, []

        # Pad / truncate to CViT2's native frame count
        if len(frames) > NUM_FRAMES:
            idx = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=int)
            frames = [frames[i] for i in idx]
        elif len(frames) < NUM_FRAMES:
            while len(frames) < NUM_FRAMES:
                frames.append(frames[-1])

        processed = [self._preprocess_frame(f) for f in frames]
        batch = torch.stack(processed).to(self.device)  # [NUM_FRAMES, 3, 224, 224]

        with torch.no_grad():
            logits = self.model(batch)  # [NUM_FRAMES, 2]
            probs = torch.sigmoid(logits).cpu().numpy()

        # Index 0 = FAKE, matching the official repo's real_or_fake() convention.
        frame_scores = probs[:, 0].tolist()

        # CViT2 classifies each frame independently, so these are genuine per-frame scores.
        avg_score = float(np.mean(frame_scores))
        return avg_score, frame_scores

    def get_verdict(self, score):
        return "Real" if score < 0.5 else "Deepfake"


if __name__ == "__main__":
    import sys
    from frame_extractor import extract_frames
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        scorer = CViTScorer()
        frames = extract_frames(video_path, num_frames=NUM_FRAMES)
        avg, per_frame = scorer.score_video_frames(frames)
        print(f"Average fake probability: {avg:.4f} -> {scorer.get_verdict(avg)}")
        print(f"Per-frame scores: {[round(s, 3) for s in per_frame]}")
