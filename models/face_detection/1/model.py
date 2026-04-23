"""
Triton Python Backend — SCRFD Face Detection Model
===================================================
Model: SCRFD-10GF with BN and keypoints (scrfd_10g_bnkps.onnx)
Backend: Python (custom preprocessing + postprocessing)
Runtime: cv2.dnn with ONNX graph execution

SCRFD Output Format (9 output heads for 3 strides: 8, 16, 32)
--------------------------------------------------------------
For each stride S in [8, 16, 32]:
  score_stride_S  : shape (1, H*W*num_anchors, 1)   — classification score (sigmoid)
  bbox_stride_S   : shape (1, H*W*num_anchors, 4)   — ltrb distances from anchor centre
  kps_stride_S    : shape (1, H*W*num_anchors, 10)  — keypoint offsets (5 points × xy)

Anchor Configuration (InsightFace SCRFD standard):
  num_anchors per location = 2
  strides = [8, 16, 32]
  anchor sizes are uniform (1×1) — positions encode spatial location only
"""

import base64
import numpy as np
import cv2
import os
import triton_python_backend_utils as pb_utils


# ── Constants ──────────────────────────────────────────────────────────────────
INPUT_SIZE   = (640, 640)   # (W, H) expected by SCRFD
NUM_ANCHORS  = 2            # anchors per feature-map location
STRIDES      = [8, 16, 32]
FEAT_STRIDE_FPNS = [8, 16, 32]


# ── Anchor generation ──────────────────────────────────────────────────────────
def _generate_anchors(stride: int, feat_h: int, feat_w: int) -> np.ndarray:
    """
    Generate anchor centre points for one feature-map stride.

    Returns
    -------
    np.ndarray  shape (feat_h * feat_w * NUM_ANCHORS, 2)  — (cx, cy) pairs
    """
    cx = np.arange(feat_w, dtype=np.float32) * stride
    cy = np.arange(feat_h, dtype=np.float32) * stride
    cx_grid, cy_grid = np.meshgrid(cx, cy)
    centres = np.stack([cx_grid.ravel(), cy_grid.ravel()], axis=1)  # (H*W, 2)
    # Repeat for num_anchors
    centres = np.repeat(centres, NUM_ANCHORS, axis=0)               # (H*W*A, 2)
    return centres


# ── Distance → bounding box ────────────────────────────────────────────────────
def _distance_to_bbox(anchor_centres: np.ndarray,
                      distances: np.ndarray) -> np.ndarray:
    """
    Decode ltrb anchor-relative distances into absolute (x1, y1, x2, y2).

    Parameters
    ----------
    anchor_centres : (N, 2)  cx, cy
    distances      : (N, 4)  left, top, right, bottom distances from centre
    """
    x1 = anchor_centres[:, 0] - distances[:, 0]
    y1 = anchor_centres[:, 1] - distances[:, 1]
    x2 = anchor_centres[:, 0] + distances[:, 2]
    y2 = anchor_centres[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)


# ── NMS ────────────────────────────────────────────────────────────────────────
def _nms(boxes: np.ndarray,
         scores: np.ndarray,
         score_threshold: float = 0.4,
         iou_threshold: float = 0.45) -> np.ndarray:
    """
    Apply Non-Maximum Suppression using cv2.dnn.NMSBoxes.

    Returns
    -------
    np.ndarray  shape (K, 5)  — [x1, y1, x2, y2, score] for kept detections
    """
    if len(boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    # cv2 NMS expects [x, y, w, h]
    cv2_boxes = [
        [float(b[0]), float(b[1]),
         float(b[2] - b[0]), float(b[3] - b[1])]
        for b in boxes
    ]
    cv2_scores = [float(s) for s in scores]

    indices = cv2.dnn.NMSBoxes(
        cv2_boxes,
        cv2_scores,
        score_threshold=score_threshold,
        nms_threshold=iou_threshold,
    )

    if len(indices) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    indices = indices.flatten()
    kept = np.hstack([
        boxes[indices],
        scores[indices].reshape(-1, 1)
    ]).astype(np.float32)

    return kept


# ── TritonPythonModel ──────────────────────────────────────────────────────────
class TritonPythonModel:

    def initialize(self, args: dict) -> None:
        """Load ONNX model via cv2.dnn at server startup."""
        model_path = os.path.join(os.path.dirname(__file__), "model.onnx")

        self.net           = cv2.dnn.readNetFromONNX(model_path)
        self.input_size    = INPUT_SIZE       # (W, H)
        self.conf_thresh   = 0.4
        self.iou_thresh    = 0.45

        # Pre-compute per-stride mean pixel values for normalisation
        self.mean  = np.array([127.5, 127.5, 127.5], dtype=np.float32)
        self.scale = 1.0 / 128.0

        print("[SmartVision-X] SCRFD model loaded from:", model_path, flush=True)

    # ── Preprocessing ──────────────────────────────────────────────────────────
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Resize → normalise → CHW → add batch dim.

        Returns
        -------
        np.ndarray  shape (1, 3, 640, 640)  float32
        """
        img = cv2.resize(image, self.input_size)                 # (H, W, 3)
        img = (img.astype(np.float32) - self.mean) * self.scale  # normalise
        img = img.transpose(2, 0, 1)                             # HWC → CHW
        img = np.expand_dims(img, axis=0)                        # add batch
        return img

    # ── SCRFD output parsing ───────────────────────────────────────────────────
    def _decode_outputs(self,
                        outputs: list,
                        orig_h: int,
                        orig_w: int) -> np.ndarray:
        """
        Decode SCRFD multi-scale outputs into bounding boxes.

        SCRFD-10GF outputs 9 heads ordered as:
          [score_8, bbox_8, kps_8, score_16, bbox_16, kps_16, score_32, bbox_32, kps_32]

        Falls back to a safe single-output path if the model was exported
        with post-processing included (7 or fewer heads).

        Returns
        -------
        np.ndarray  shape (K, 5)  — [x1, y1, x2, y2, score] after NMS
        """
        feat_h = self.input_size[1]
        feat_w = self.input_size[0]

        scale_x = orig_w / feat_w
        scale_y = orig_h / feat_h

        all_boxes  = []
        all_scores = []

        # ── Multi-head SCRFD path (9 outputs expected) ─────────────────────────
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 6:
            # Group outputs: (score, bbox, kps) × stride
            # Outputs order from InsightFace SCRFD export (stride ascending):
            #   score_8, bbox_8, kps_8, score_16, bbox_16, kps_16, score_32, bbox_32, kps_32
            # If kps heads are absent (6 outputs): score_8, bbox_8, score_16, bbox_16, ...
            has_kps     = len(outputs) == 9
            group_size  = 3 if has_kps else 2

            for idx, stride in enumerate(STRIDES):
                base        = idx * group_size
                score_head  = outputs[base]        # (1, H*W*A, 1) or (1, H*W*A)
                bbox_head   = outputs[base + 1]    # (1, H*W*A, 4)

                score_head = score_head.reshape(-1)          # flatten to (N,)
                bbox_head  = bbox_head.reshape(-1, 4)        # (N, 4)

                # Sigmoid for scores
                scores_sig = 1.0 / (1.0 + np.exp(-score_head))

                # Filter by confidence
                keep = scores_sig >= self.conf_thresh
                if not keep.any():
                    continue

                scores_sig = scores_sig[keep]
                bbox_head  = bbox_head[keep]

                # Generate anchors for this stride
                f_h       = feat_h // stride
                f_w       = feat_w // stride
                anchors   = _generate_anchors(stride, f_h, f_w)
                anchors   = anchors[keep]

                # Decode: distances → absolute boxes at input resolution
                boxes_abs = _distance_to_bbox(anchors, bbox_head * stride)

                # Scale to original image resolution
                boxes_abs[:, 0] *= scale_x
                boxes_abs[:, 1] *= scale_y
                boxes_abs[:, 2] *= scale_x
                boxes_abs[:, 3] *= scale_y

                all_boxes.append(boxes_abs)
                all_scores.append(scores_sig)

        # ── Fallback: single or flat output (post-processed model) ─────────────
        else:
            raw = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            raw = raw.squeeze()

            if raw.ndim == 1:
                raw = raw.reshape(1, -1)

            for row in raw:
                if len(row) < 5:
                    continue
                score = float(row[4])
                if score < self.conf_thresh:
                    continue
                x1 = float(row[0]) * scale_x
                y1 = float(row[1]) * scale_y
                x2 = float(row[2]) * scale_x
                y2 = float(row[3]) * scale_y
                if x2 <= x1 or y2 <= y1:
                    continue
                all_boxes.append(np.array([[x1, y1, x2, y2]], dtype=np.float32))
                all_scores.append(np.array([score], dtype=np.float32))

        if not all_boxes:
            return np.zeros((0, 5), dtype=np.float32)

        all_boxes  = np.concatenate(all_boxes,  axis=0)
        all_scores = np.concatenate(all_scores, axis=0)

        # Clip to image boundaries
        all_boxes[:, 0::2] = np.clip(all_boxes[:, 0::2], 0, orig_w)
        all_boxes[:, 1::2] = np.clip(all_boxes[:, 1::2], 0, orig_h)

        # Remove degenerate boxes
        valid = (all_boxes[:, 2] > all_boxes[:, 0]) & \
                (all_boxes[:, 3] > all_boxes[:, 1])
        all_boxes  = all_boxes[valid]
        all_scores = all_scores[valid]

        return _nms(all_boxes, all_scores,
                    score_threshold=self.conf_thresh,
                    iou_threshold=self.iou_thresh)

    # ── Triton execute ─────────────────────────────────────────────────────────
    def execute(self, requests: list) -> list:
        responses = []

        for request in requests:
            # ── Receive input tensor (TYPE_STRING carries raw base64 bytes) ────
            input_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE")
            data = input_tensor.as_numpy()

            # data[0] is the base64-encoded JPEG string sent by face_detector.py
            try:
                raw_bytes = base64.b64decode(data[0])
            except Exception as exc:
                print(f"[SCRFD] base64 decode failed: {exc}", flush=True)
                empty = np.zeros((0, 5), dtype=np.float32)
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("BBOXES", empty)]
                ))
                continue

            image = cv2.imdecode(
                np.frombuffer(raw_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                print("[SCRFD] cv2.imdecode returned None — invalid image bytes",
                      flush=True)
                empty = np.zeros((0, 5), dtype=np.float32)
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("BBOXES", empty)]
                ))
                continue

            orig_h, orig_w = image.shape[:2]

            # ── Preprocess ────────────────────────────────────────────────────
            blob = self._preprocess(image)

            # ── Run ONNX via cv2.dnn ──────────────────────────────────────────
            self.net.setInput(blob)
            outputs = self.net.forward(
                self.net.getUnconnectedOutLayersNames()
            )

            print(
                f"[SCRFD] raw output count={len(outputs) if isinstance(outputs, list) else 1}",
                flush=True,
            )

            # ── Decode + NMS ──────────────────────────────────────────────────
            boxes = self._decode_outputs(outputs, orig_h, orig_w)

            print(f"[SCRFD] faces after NMS: {len(boxes)}", flush=True)

            out_tensor = pb_utils.Tensor("BBOXES", boxes)
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[out_tensor]
            ))

        return responses
