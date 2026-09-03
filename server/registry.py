"""
In-memory registry of connected device workers.
Thread-safe for asyncio (single-threaded event loop).
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

TOTAL_LAYERS = 12  # GPT-2 Small


@dataclass
class DeviceInfo:
    device_id:   str
    url:         str
    layer_start: int
    layer_end:   int
    is_first:    bool
    is_last:     bool
    status:      str   = "ready"
    registered:  float = field(default_factory=time.time)


class DeviceRegistry:
    def __init__(self):
        self._store: Dict[str, DeviceInfo] = {}

    def register(self, data: dict) -> dict:
        dev = DeviceInfo(**data)
        self._store[dev.device_id] = dev
        return {"registered": dev.device_id, "layers": [dev.layer_start, dev.layer_end]}

    def mark_unavailable(self, device_id: str) -> None:
        if device_id in self._store:
            self._store[device_id].status = "unavailable"

    def all_devices(self) -> List[dict]:
        return [
            {
                "device_id":   d.device_id,
                "layers":      [d.layer_start, d.layer_end],
                "status":      d.status,
                "is_first":    d.is_first,
                "is_last":     d.is_last,
            }
            for d in self._store.values()
        ]

    def get_pipeline(self) -> Optional[List[DeviceInfo]]:
        """
        Returns an ordered list of READY devices that together cover
        layers 0..TOTAL_LAYERS contiguously, or None if not ready.
        """
        ready = [d for d in self._store.values() if d.status == "ready"]
        if not ready:
            return None

        # Sort by layer_start; verify contiguous coverage 0 -> TOTAL_LAYERS
        ordered = sorted(ready, key=lambda d: d.layer_start)
        expected = 0
        for dev in ordered:
            if dev.layer_start != expected:
                return None
            expected = dev.layer_end

        if expected != TOTAL_LAYERS:
            return None

        # Must have exactly one first and one last
        if sum(1 for d in ordered if d.is_first) != 1:
            return None
        if sum(1 for d in ordered if d.is_last) != 1:
            return None

        return ordered
