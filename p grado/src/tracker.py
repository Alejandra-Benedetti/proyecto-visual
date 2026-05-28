"""
Cross-image object tracking using ByteTrack via supervision.
Images are processed as sequential "frames" to maintain consistent
object IDs across photos taken at the same location.
"""
import numpy as np
from pathlib import Path


class ObjectTracker:
    def __init__(self):
        self._tracker = None
        self.history: dict[int, list] = {}

    @property
    def tracker(self):
        if self._tracker is None:
            import supervision as sv
            self._tracker = sv.ByteTrack()
        return self._tracker

    def update(self, det: dict, image_path: str | Path) -> list:
        import supervision as sv

        dets = det.get("detections", [])
        if not dets:
            return []

        xyxy = np.array([d["bbox"] for d in dets], dtype=np.float32)
        conf = np.array([d["confidence"] for d in dets], dtype=np.float32)
        cls_id = np.array([d["class_id"] for d in dets], dtype=int)

        sv_det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls_id)
        tracked = self.tracker.update_with_detections(sv_det)

        tracks = []
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            entry = {
                "image": str(image_path),
                "bbox": tracked.xyxy[i].tolist(),
                "class_id": int(tracked.class_id[i]),
                "confidence": float(tracked.confidence[i]),
            }
            self.history.setdefault(tid, []).append(entry)
            tracks.append({"track_id": tid, **entry})

        return tracks

    def persistent_objects(self, min_appearances: int = 2) -> dict:
        return {
            tid: app
            for tid, app in self.history.items()
            if len(app) >= min_appearances
        }

    def reset(self):
        import supervision as sv
        self._tracker = sv.ByteTrack()
        self.history = {}
