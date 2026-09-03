"""
Room management — creates rooms, assigns model slice slots to devices.

Fixed 3-slice model:
  slot 0: layers  0-4  (embedding + first 4 transformer blocks)
  slot 1: layers  4-8
  slot 2: layers 8-12  (last 4 blocks + LM head)

Browser workers store a WebSocket reference in `dev.ws`.
Python workers store their HTTP URL in `dev.url`.
"""
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SLICE_ASSIGNMENTS = [(0, 4), (4, 8), (8, 12)]
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
    room_id: str
    slots:   Dict[int, Optional[RoomDevice]] = field(
        default_factory=lambda: {i: None for i in range(NUM_SLICES)}
    )

    def join(self, device_id: str, ram_mb: int) -> Optional[RoomDevice]:
        """Return existing assignment if device already joined, else assign next free slot."""
        for dev in self.slots.values():
            if dev and dev.device_id == device_id:
                return dev

        for slot_id in range(NUM_SLICES):
            if self.slots[slot_id] is None:
                layer_start, layer_end = SLICE_ASSIGNMENTS[slot_id]
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

    def to_dict(self) -> dict:
        return {
            "room_id":        self.room_id,
            "devices":        [d.to_dict() for d in self.slots.values() if d is not None],
            "pipeline_ready": self.pipeline_ready,
            "ready_count":    self.ready_count(),
            "total_slots":    NUM_SLICES,
        }


class RoomManager:
    def __init__(self):
        self._rooms: Dict[str, Room] = {}

    def create(self) -> Room:
        room_id = uuid.uuid4().hex[:8]
        room = Room(room_id=room_id)
        self._rooms[room_id] = room
        return room

    def get(self, room_id: str) -> Optional[Room]:
        return self._rooms.get(room_id)

    def get_or_create(self, room_id: str) -> Room:
        if room_id not in self._rooms:
            self._rooms[room_id] = Room(room_id=room_id)
        return self._rooms[room_id]
