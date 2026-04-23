"""
SmartVision-X — Model Download Utility
=======================================
Downloads SCRFD-10GF face detection model weights from the configured
model repository into the project root for use by Triton.

Usage:
    python download_model.py

Authentication:
    Token is read from infra/token_Smartvision-X.txt
    This file must exist locally and must NOT be committed to version control.
"""

import os
import sys

TOKEN_FILE  = os.path.join("infra", "token_Smartvision-X.txt")
OUTPUT_PATH = "models/face_detection/1/model.onnx"


def load_token(path: str) -> str:
    if not os.path.exists(path):
        print(f"[ERROR] Token file not found: {path}")
        print("        Create the file and add your access token.")
        sys.exit(1)
    token = open(path).read().strip()
    if not token:
        print(f"[ERROR] Token file is empty: {path}")
        sys.exit(1)
    return token


def download_model(token: str, output_path: str) -> None:
    """
    Download the SCRFD model weights.
    Replace the URL and download logic with your actual model repository.
    """
    try:
        import requests
    except ImportError:
        print("[ERROR] 'requests' library not installed. Run: pip install requests")
        sys.exit(1)

    # ── Replace with your actual model repository URL ──────────────────────
    MODEL_URL = "https://your-model-repo/scrfd_10g_bnkps.onnx"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[INFO]  Downloading SCRFD model to {output_path} …")
    response = requests.get(
        MODEL_URL,
        headers={"Authorization": f"token {token}"},
        stream=True,
        timeout=120,
    )

    if response.status_code == 401:
        print("[ERROR] Authentication failed — check your token")
        sys.exit(1)

    if response.status_code != 200:
        print(f"[ERROR] Download failed with HTTP {response.status_code}")
        sys.exit(1)

    total = int(response.headers.get("content-length", 0))
    received = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            received += len(chunk)
            if total:
                pct = received / total * 100
                print(f"\r[INFO]  {pct:.1f}%  ({received // 1024 // 1024} MB)", end="", flush=True)

    print(f"\n[OK]    Model saved to {output_path}")


if __name__ == "__main__":
    token = load_token(TOKEN_FILE)
    download_model(token, OUTPUT_PATH)
