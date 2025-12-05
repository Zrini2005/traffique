"""Simple online tracker: motion-only greedy assignment with EMA embedding.

This is a small, dependency-light tracker intended to be a drop-in that
provides the same API shape the server expects during tests. It uses a
nearest-center greedy match and an exponential moving average for the
appearance vector if provided.
"""
from typing import List, Dict, Any, Tuple
import numpy as np


class Track:
    def __init__(self, track_id: int, init_bbox: Tuple[int, int, int, int], frame_idx: int, embedding: np.ndarray = None):
        self.id = track_id
        self.bboxes = [init_bbox]
        self.frames = [frame_idx]
        self.embedding = embedding.copy() if embedding is not None else None
        self.missed = 0

    def last_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bboxes[-1]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, bbox: Tuple[int, int, int, int], frame_idx: int, embedding: np.ndarray = None, ema_alpha: float = 0.2):
        self.bboxes.append(bbox)
        self.frames.append(frame_idx)
        self.missed = 0
        if embedding is not None:
            if self.embedding is None:
                self.embedding = embedding.copy()
            else:
                # EMA
                self.embedding = (1 - ema_alpha) * self.embedding + ema_alpha * embedding


class OnlineTracker:
    def __init__(self, max_missed: int = 30, dist_thresh_px: float = 120.0, appearance_weight: float = 0.6, ema_alpha: float = 0.2):
        """Simple online tracker.

        Args:
            max_missed: how many frames to tolerate missing before pruning
            dist_thresh_px: gating distance in pixels for motion
            appearance_weight: weight given to appearance (0..1) when combining costs
            ema_alpha: EMA alpha for updating per-track embedding
        """
        self.dist_thresh = dist_thresh_px
        self.max_missed = max_missed
        self.ema_alpha = ema_alpha
        self.appearance_weight = float(appearance_weight)
        self.next_id = 1
        self.tracks: List[Track] = []

    def _center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, detections: List[Dict[str, Any]], frame_idx: int):
        """Update tracks with detections.

        detections: list of dicts with 'bbox' and optional 'embedding'
        """
        det_centers = [self._center(d['bbox']) for d in detections]
        det_embs = [d.get('embedding') for d in detections]
        assigned = set()

        # Greedy assignment using combined motion + appearance score
        for t in self.tracks:
            if len(det_centers) == 0:
                t.missed += 1
                continue
            best_i = None
            best_score = float('inf')
            tc = t.last_center()
            for i, dc in enumerate(det_centers):
                if i in assigned:
                    continue
                # pixel distance squared
                pd2 = (tc[0] - dc[0]) ** 2 + (tc[1] - dc[1]) ** 2
                if pd2 > (self.dist_thresh ** 2):
                    # outside gating, skip
                    continue

                # appearance distance (cosine) if available
                app_dist = 0.0
                te = t.embedding
                de = det_embs[i]
                if te is not None and de is not None and te.size == de.size and np.linalg.norm(te) > 0 and np.linalg.norm(de) > 0:
                    # embeddings assumed L2-normalized; cosine dist = 1 - dot
                    app_dist = 1.0 - float(np.dot(te.ravel(), de.ravel()))
                    # clamp
                    app_dist = max(0.0, min(2.0, app_dist))

                # normalize pixel distance to 0..1 by dist_thresh^2
                motion_cost = float(pd2) / float(max(1.0, (self.dist_thresh ** 2)))

                # combined cost (lower is better)
                score = (1.0 - self.appearance_weight) * motion_cost + self.appearance_weight * app_dist

                if score < best_score:
                    best_score = score
                    best_i = i

            if best_i is not None:
                det = detections[best_i]
                t.update(det['bbox'], frame_idx, embedding=det.get('embedding'), ema_alpha=self.ema_alpha)
                assigned.add(best_i)
            else:
                t.missed += 1

        # Create tracks for unassigned detections
        for i, det in enumerate(detections):
            if i in assigned:
                continue
            tr = Track(self.next_id, det['bbox'], frame_idx, embedding=det.get('embedding'))
            self.next_id += 1
            self.tracks.append(tr)

        # Prune dead tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

    def get_active_tracks(self, min_length: int = 1) -> List[Dict[str, Any]]:
        out = []
        for t in self.tracks:
            if len(t.bboxes) >= min_length:
                out.append({
                    'track_id': t.id,
                    'bboxes': t.bboxes,
                    'frames': t.frames,
                    'embedding': t.embedding
                })
        return out


if __name__ == "__main__":
    # smoke test
    ot = OnlineTracker()
    dets = [{'bbox': (10, 10, 50, 60)}, {'bbox': (200, 30, 240, 70)}]
    ot.update(dets, 0)
    ot.update([{'bbox': (12, 12, 52, 62)}], 1)
    print(ot.get_active_tracks())
