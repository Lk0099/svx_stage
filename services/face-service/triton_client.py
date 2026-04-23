"""
Triton Inference Server client for SmartVision-X face recognition pipeline.

Provides:
  - infer_face_recognition(face_input) → embedding vector (np.ndarray)

Connects to Triton at http://triton:8000 via the tritonclient HTTP library.
Tensor names and shapes must match models/face_recognition/config.pbtxt exactly.
"""

import numpy as np
import tritonclient.http as httpclient

# ── Triton connection ──────────────────────────────────────────────────────────
TRITON_URL        = "triton:8000"
RECOGNITION_MODEL = "face_recognition"

# Input/output names must match config.pbtxt exactly
INPUT_NAME  = "input.1"
OUTPUT_NAME = "683"


def _get_client() -> httpclient.InferenceServerClient:
    """Create a Triton HTTP client.  Called per-request (lightweight object)."""
    return httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)


def infer_face_recognition(face_input: np.ndarray) -> np.ndarray:
    """
    Run face recognition inference on Triton.

    Parameters
    ----------
    face_input : np.ndarray
        Preprocessed face crop with shape (1, 3, 112, 112) and dtype float32.
        The caller (main.py) is responsible for resizing, channel transposing,
        normalising, and adding the batch dimension before calling this function.

    Returns
    -------
    np.ndarray
        Raw embedding vector of shape (1, 512) and dtype float32.
        The caller is responsible for L2 normalisation before storage or comparison.

    Raises
    ------
    ValueError
        If face_input has an unexpected shape or dtype.
    RuntimeError
        If the Triton inference call fails.
    """
    # ── Input validation ───────────────────────────────────────────────────────
    if face_input is None:
        raise ValueError("face_input must not be None")

    if face_input.ndim != 4 or face_input.shape != (1, 3, 112, 112):
        raise ValueError(
            f"Expected face_input shape (1, 3, 112, 112), got {face_input.shape}"
        )

    # Enforce dtype — config.pbtxt declares TYPE_FP32
    face_input = face_input.astype(np.float32)

    # ── Build input tensor ─────────────────────────────────────────────────────
    infer_input = httpclient.InferInput(
        name=INPUT_NAME,
        shape=list(face_input.shape),   # [1, 3, 112, 112]
        datatype="FP32",
    )
    infer_input.set_data_from_numpy(face_input, binary_data=True)

    # ── Build requested output ─────────────────────────────────────────────────
    infer_output = httpclient.InferRequestedOutput(
        name=OUTPUT_NAME,
        binary_data=True,
    )

    # ── Run inference ──────────────────────────────────────────────────────────
    try:
        client   = _get_client()
        response = client.infer(
            model_name=RECOGNITION_MODEL,
            inputs=[infer_input],
            outputs=[infer_output],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Triton inference failed for model '{RECOGNITION_MODEL}': {exc}"
        ) from exc

    # ── Parse output ───────────────────────────────────────────────────────────
    embedding = response.as_numpy(OUTPUT_NAME)  # shape: (1, 512), dtype: float32

    if embedding is None:
        raise RuntimeError(
            f"Triton returned no data for output tensor '{OUTPUT_NAME}'"
        )

    return embedding
