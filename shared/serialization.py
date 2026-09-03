import base64
import numpy as np


def tensor_to_payload(arr: np.ndarray) -> dict:
    """Encode a float array to a JSON-serialisable dict (float16 bytes + shape)."""
    arr = np.asarray(arr, dtype=np.float16)
    return {
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
    }


def payload_to_tensor(payload: dict) -> np.ndarray:
    """Decode a dict produced by tensor_to_payload back to a float32 ndarray."""
    raw = base64.b64decode(payload["data"])
    return np.frombuffer(raw, dtype=np.float16).reshape(payload["shape"]).astype(np.float32).copy()
