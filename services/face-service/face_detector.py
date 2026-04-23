"""
FaceDetector
============
Sends an image to the SCRFD face detection model on Triton Inference Server
via the HTTP v2 inference protocol and returns a list of bounding boxes.

The image is base64-encoded before transmission.  The Triton Python backend
(models/face_detection/1/model.py) decodes it with base64.b64decode() before
running SCRFD inference.
"""

import base64
import json

import cv2
import numpy as np
import requests


class FaceDetector:

    TRITON_URL   = "http://triton:8000/v2/models/face_detection/infer"
    TIMEOUT_SEC  = 30

    def __init__(self, conf_threshold: float = 0.5) -> None:
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect faces in a BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR image array (H, W, 3) uint8 as returned by cv2.imdecode.

        Returns
        -------
        list of (x1, y1, x2, y2) tuples in pixel coordinates,
        sorted by confidence descending.  Returns [] on any failure.
        """
        if image is None or image.size == 0:
            return []

        # ── Encode image → JPEG bytes → base64 string ─────────────────────────
        success, img_encoded = cv2.imencode(".jpg", image)
        if not success:
            print("[FaceDetector] cv2.imencode failed", flush=True)
            return []

        img_b64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

        # ── Build Triton HTTP v2 inference request ─────────────────────────────
        payload = {
            "inputs": [
                {
                    "name":     "IMAGE",
                    "shape":    [1],
                    "datatype": "BYTES",
                    "data":     [img_b64],
                }
            ]
        }

        # ── Call Triton ────────────────────────────────────────────────────────
        try:
            response = requests.post(
                self.TRITON_URL,
                json=payload,
                timeout=self.TIMEOUT_SEC,
            )
        except requests.exceptions.ConnectionError:
            print("[FaceDetector] Cannot connect to Triton — is the container running?",
                  flush=True)
            return []
        except requests.exceptions.Timeout:
            print("[FaceDetector] Triton request timed out", flush=True)
            return []
        except Exception as exc:
            print(f"[FaceDetector] Unexpected request error: {exc}", flush=True)
            return []

        if response.status_code != 200:
            print(
                f"[FaceDetector] Triton returned HTTP {response.status_code}: "
                f"{response.text[:200]}",
                flush=True,
            )
            return []

        # ── Parse response ─────────────────────────────────────────────────────
        try:
            body    = response.json()
            outputs = body["outputs"]
            # Find BBOXES output
            bbox_out = next(
                (o for o in outputs if o["name"] == "BBOXES"), None
            )
            if bbox_out is None:
                print("[FaceDetector] 'BBOXES' not found in Triton response", flush=True)
                return []
            data = bbox_out["data"]
        except (KeyError, json.JSONDecodeError, StopIteration) as exc:
            print(f"[FaceDetector] Failed to parse Triton response: {exc}", flush=True)
            return []

        if not data:
            return []

        # ── Convert to list of (x1, y1, x2, y2), filter by threshold ──────────
        try:
            boxes_raw = np.array(data, dtype=np.float32).reshape(-1, 5)
        except ValueError as exc:
            print(f"[FaceDetector] Cannot reshape BBOXES data: {exc}", flush=True)
            return []

        faces = []
        for row in boxes_raw:
            x1, y1, x2, y2, score = row
            if score < self.conf_threshold:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            faces.append((int(x1), int(y1), int(x2), int(y2)))

        # Sort by detection area descending (largest / most prominent face first)
        faces.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

        print(f"[FaceDetector] {len(faces)} face(s) detected above threshold "
              f"{self.conf_threshold}", flush=True)

        return faces
