import sys

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        __import__(import_name)
        print(f"✅ {name} is installed")
    except ImportError:
        print(f"❌ {name} is NOT installed")
        return False
    return True

print("Checking packages...")
# Every runtime dependency (the first section of requirements.txt). Checking
# only a subset used to
# let a machine pass here and still fail at runtime on Signals 2, 3 or 4, or on
# the social-media URL pipeline (yt-dlp).
packages = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("opencv-python-headless", "cv2"),
    ("Pillow", "PIL"),
    ("numpy", "numpy"),
    ("Flask", "flask"),
    ("scikit-learn", "sklearn"),
    ("Werkzeug", "werkzeug"),
    ("einops", "einops"),
    ("huggingface_hub", "huggingface_hub"),
    ("soundfile", "soundfile"),
    ("librosa", "librosa"),
    ("sentence-transformers", "sentence_transformers"),
    ("reportlab", "reportlab"),
    ("yt-dlp", "yt_dlp"),
]

all_passed = True
for name, import_name in packages:
    if not check_package(name, import_name):
        all_passed = False

print("\nChecking system binaries...")
import shutil
if shutil.which("ffmpeg"):
    print("\u2705 ffmpeg is on PATH")
else:
    print("\u274c ffmpeg is NOT on PATH - Signals 2 and 3 cannot decode audio")
    all_passed = False

print("\nChecking PyTorch Device...")
try:
    import torch
    if torch.backends.mps.is_available():
        print("✅ Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        print("✅ Device: CUDA (GPU)")
    else:
        print("✅ Device: CPU")
except ImportError:
    print("❌ Cannot check device because torch is not installed.")

if all_passed:
    print("\n✅ All checks passed successfully.")
else:
    print("\n❌ Some checks failed. Please review the output above.")
    sys.exit(1)
