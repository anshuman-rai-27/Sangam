import base64
import numpy as np
import torch


def tensor_to_payload(tensor: torch.Tensor) -> dict:
    """Encode a float tensor to a JSON-serialisable dict (float16 bytes + shape)."""
    arr = tensor.detach().cpu().float().numpy().astype(np.float16)
    return {
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
    }


def payload_to_tensor(payload: dict) -> torch.Tensor:
    """Decode a dict produced by tensor_to_payload back to a float32 torch.Tensor."""
    raw = base64.b64decode(payload["data"])
    arr = np.frombuffer(raw, dtype=np.float16).reshape(payload["shape"]).astype(np.float32)
    return torch.from_numpy(arr.copy())
