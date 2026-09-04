"""
Room management — creates rooms, assigns model slice slots to devices.

slice_assignments come from the model config (e.g. Qwen2.5: 8 layers/slice,
GPT-2: 4 layers/slice). Browser workers store a WebSocket ref in dev.ws;
Python workers store their HTTP URL in dev.url.
"""
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

NUM_SLICES = 3


@dataclass
class RoomDevice:
    device_id:   str
    ram_mb:      int
    slice_id:    int
    layer_start: int
    layer_end:   int
    status:      str = "assigned"   # assigned | ready | dropped
    url:         str = ""           # Python worker HTTP URL
    ws:          Any = field(default=None, repr=False)  # browser worker WebSocket

    @property
    def is_first(self) -> bool:
        return self.slice_id == 0

    @property
    def is_last(self) -> bool:
        return self.slice_id == NUM_SLICES - 1

    def to_dict(self) -> dict:
        return {
            "device_id":  self.device_id,
            "slice_id":   self.slice_id,
            "layers":     [self.layer_start, self.layer_end],
            "status":     self.status,
            "url":        self.url,
            "is_browser": self.ws is not None,
        }


@dataclass
class Room:
    room_id:           str
    model:             str = "qwen2.5-0.5b"
    model_name:        str = "Qwen2.5-0.5B-Instruct"
    slice_assignments: List[Tuple[int, int]] = field(
        default_factory=lambda: [(0, 8), (8, 16), (16, 24)]
    )
    slots: Dict[int, Optional[RoomDevice]] = field(
        default_factory=lambda: {i: None for i in range(NUM_SLICES)}
    )

    def join(self, device_id: str, ram_mb: int) -> Optional[RoomDevice]:
        """Return existing assignment if device already joined, else assign next free slot."""
        for dev in self.slots.values():
            if dev and dev.device_id == device_id:
                return dev

        for slot_id in range(NUM_SLICES):
            if self.slots[slot_id] is None:
                layer_start, layer_end = self.slice_assignments[slot_id]
                dev = RoomDevice(
                    device_id=device_id,
                    ram_mb=ram_mb,
                    slice_id=slot_id,
                    layer_start=layer_start,
                    layer_end=layer_end,
                )
                self.slots[slot_id] = dev
                return dev

        return None  # room full

    def get_device(self, device_id: str) -> Optional[RoomDevice]:
        for dev in self.slots.values():
            if dev and dev.device_id == device_id:
                return dev
        return None

    def mark_ready(self, device_id: str, worker_url: str) -> bool:
        for dev in self.slots.values():
            if dev and dev.device_id == device_id:
                dev.status = "ready"
                dev.url = worker_url
                return True
        return False

    def mark_dropped(self, device_id: str) -> bool:
        for dev in self.slots.values():
            if dev and dev.device_id == device_id:
                dev.status = "dropped"
                dev.ws = None
                return True
        return False

    @property
    def pipeline_ready(self) -> bool:
        return all(
            self.slots[i] is not None and self.slots[i].status == "ready"
            for i in range(NUM_SLICES)
        )

    def get_pipeline(self) -> Optional[List[RoomDevice]]:
        if not self.pipeline_ready:
            return None
        return [self.slots[i] for i in range(NUM_SLICES)]

    def ready_count(self) -> int:
        return sum(1 for d in self.slots.values() if d and d.status == "ready")

    def slice_labels(self) -> List[str]:
        labels = []
        for i, (start, end) in enumerate(self.slice_assignments):
            suffix = " + norm" if i == NUM_SLICES - 1 else ""
            labels.append(f"layers {start}–{end - 1}{suffix}")
        return labels

    def to_dict(self) -> dict:
        return {
            "room_id":        self.room_id,
            "model":          self.model,
            "model_name":     self.model_name,
            "slice_labels":   self.slice_labels(),
            "devices":        [d.to_dict() for d in self.slots.values() if d is not None],
            "pipeline_ready": self.pipeline_ready,
            "ready_count":    self.ready_count(),
            "total_slots":    NUM_SLICES,
        }


class RoomManager:
    def __init__(self):
        self._rooms: Dict[str, Room] = {}

    def create(self, model: str = "qwen2.5-0.5b", model_name: str = "Qwen2.5-0.5B-Instruct",
               slice_assignments: Optional[List[Tuple[int, int]]] = None) -> Room:
        if slice_assignments is None:
            slice_assignments = [(0, 8), (8, 16), (16, 24)]
        room_id = uuid.uuid4().hex[:8]
        room = Room(room_id=room_id, model=model, model_name=model_name,
                    slice_assignments=slice_assignments)
        self._rooms[room_id] = room
        return room

    def get(self, room_id: str) -> Optional[Room]:
        return self._rooms.get(room_id)

    def get_or_create(self, room_id: str) -> Room:
        if room_id not in self._rooms:
            self._rooms[room_id] = Room(room_id=room_id)
        return self._rooms[room_id]
