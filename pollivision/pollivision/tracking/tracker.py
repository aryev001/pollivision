"""Multi-object tracking with temporal cue smoothing.

Tracking earns its place here for three separate reasons, only the first of
which is the usual one:

1. **Stable identity.** The pollination ledger needs to know that the flower in
   frame 200 is the same one serviced in frame 150, or the rover will re-visit
   flowers it has already pollinated.
2. **Accuracy.** Sex, stage and pollen load are all estimated per-frame from a
   moving platform with a rolling-shutter sensor. Individual frames are noisy;
   the underlying property is constant. Averaging a cue over a track is close to
   free and recovers a substantial amount of the accuracy that motion blur,
   changing viewpoint and JPEG artefacts take away.
3. **Throughput.** With tracking in place the detector can run every Nth frame
   and the tracker bridges the gap, which is what makes a 4-core SBC viable.

The association is IoU-plus-centroid greedy matching rather than a Kalman
filter: at rover approach speeds under dense foliage, flowers move smoothly and
predictably in the image, and the extra machinery would not pay for itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..types import AnthesisStage, BBox, FlowerObservation, FlowerSex


@dataclass
class Track:
    """Temporal state for one flower."""

    track_id: int
    box: BBox
    age: int = 0              # frames since the last match
    hits: int = 1             # total matched frames
    total_frames: int = 1

    # Exponentially smoothed cue estimates.
    p_female: float = 0.5
    pollen: float = 0.0
    stage_votes: dict[str, int] = field(default_factory=dict)

    last_seen_frame: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2

    def dominant_stage(self) -> Optional[str]:
        if not self.stage_votes:
            return None
        return max(self.stage_votes, key=self.stage_votes.get)


class FlowerTracker:
    """Greedy IoU + centroid tracker with per-track cue smoothing."""

    def __init__(self, cfg) -> None:
        section = cfg.section("tracking")
        self.enabled = bool(section.get("enabled", True))
        self.max_age = int(section.get("max_age", 15))
        self.min_hits = int(section.get("min_hits", 2))
        self.iou_threshold = float(section.get("iou_threshold", 0.25))
        self.max_centre_distance = float(section.get("max_centre_distance_px", 120))

        smoothing = section.section("smoothing")
        self.sex_alpha = float(smoothing.get("sex_alpha", 0.35))
        self.pollen_alpha = float(smoothing.get("pollen_alpha", 0.4))
        self.stage_votes = int(smoothing.get("stage_votes", 5))

        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self._frame = 0

    # ------------------------------------------------------------------ #

    def update(self, flowers: list[FlowerObservation]) -> list[FlowerObservation]:
        """Assign track IDs and apply temporal smoothing, in place."""
        self._frame += 1
        if not self.enabled:
            return flowers

        matches, unmatched = self._associate(flowers)

        for flower_index, track_id in matches.items():
            self._update_track(self.tracks[track_id], flowers[flower_index])

        for flower_index in unmatched:
            track = self._create_track(flowers[flower_index])
            matches[flower_index] = track.track_id

        # Age out tracks that were not matched this frame.
        for track_id, track in list(self.tracks.items()):
            if track.last_seen_frame != self._frame:
                track.age += 1
                if track.age > self.max_age:
                    del self.tracks[track_id]

        for flower_index, track_id in matches.items():
            track = self.tracks.get(track_id)
            if track is not None:
                self._apply_smoothing(flowers[flower_index], track)

        return flowers

    # ------------------------------------------------------------------ #

    def _associate(self, flowers: list[FlowerObservation]) -> tuple[dict[int, int], list[int]]:
        """Greedy match of detections to tracks by IoU, then by centroid."""
        if not flowers:
            return {}, []
        if not self.tracks:
            return {}, list(range(len(flowers)))

        track_ids = list(self.tracks)
        cost = np.full((len(flowers), len(track_ids)), -1.0, dtype=np.float32)

        for i, flower in enumerate(flowers):
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                iou = flower.box.iou(track.box)
                if iou >= self.iou_threshold:
                    cost[i, j] = 1.0 + iou  # IoU matches always beat centroid ones
                    continue
                distance = float(np.hypot(flower.box.cx - track.box.cx,
                                          flower.box.cy - track.box.cy))
                # Fall back to proximity, scaled by object size so the same
                # threshold works for a 3 cm cucumber flower and a 10 cm pumpkin.
                limit = min(self.max_centre_distance,
                            max(flower.box.width, flower.box.height) * 1.5)
                if distance <= limit:
                    cost[i, j] = 1.0 - distance / max(limit, 1e-6)

        matches: dict[int, int] = {}
        used_tracks: set[int] = set()
        used_flowers: set[int] = set()

        # Greedy: repeatedly take the best remaining pair.
        while True:
            index = int(np.argmax(cost))
            i, j = divmod(index, cost.shape[1])
            if cost[i, j] <= 0:
                break
            matches[i] = track_ids[j]
            used_flowers.add(i)
            used_tracks.add(j)
            cost[i, :] = -1.0
            cost[:, j] = -1.0

        unmatched = [i for i in range(len(flowers)) if i not in used_flowers]
        return matches, unmatched

    def _create_track(self, flower: FlowerObservation) -> Track:
        track = Track(
            track_id=self._next_id,
            box=flower.box,
            p_female=flower.sex.p_female,
            pollen=flower.pollen.availability,
            last_seen_frame=self._frame,
        )
        stage = flower.anthesis.stage.value
        if stage != AnthesisStage.UNKNOWN.value:
            track.stage_votes[stage] = 1
        self.tracks[track.track_id] = track
        self._next_id += 1
        return track

    def _update_track(self, track: Track, flower: FlowerObservation) -> None:
        track.box = flower.box
        track.age = 0
        track.hits += 1
        track.total_frames += 1
        track.last_seen_frame = self._frame

        # Weight each frame's contribution by how confident that frame was, so a
        # blurred or half-occluded view barely moves the running estimate.
        sex_weight = self.sex_alpha * float(np.clip(flower.sex.confidence, 0.05, 1.0))
        track.p_female = (1.0 - sex_weight) * track.p_female + sex_weight * flower.sex.p_female

        pollen_weight = self.pollen_alpha * float(np.clip(flower.pollen.confidence, 0.05, 1.0))
        track.pollen = (1.0 - pollen_weight) * track.pollen + \
            pollen_weight * flower.pollen.availability

        stage = flower.anthesis.stage.value
        if stage != AnthesisStage.UNKNOWN.value:
            track.stage_votes[stage] = track.stage_votes.get(stage, 0) + 1
            # Bound the vote history so the track can still follow a real
            # transition from opening to receptive to senescent.
            total = sum(track.stage_votes.values())
            if total > self.stage_votes * 2:
                track.stage_votes = {k: max(1, v // 2) for k, v in track.stage_votes.items()}

    def _apply_smoothing(self, flower: FlowerObservation, track: Track) -> None:
        """Replace per-frame cue values with their smoothed track estimates."""
        flower.track_id = track.track_id
        flower.meta["track_hits"] = track.hits

        if not track.confirmed:
            # A brand-new track has no history worth trusting yet, so leave the
            # single-frame estimate alone rather than pretending to smooth it.
            return

        flower.sex.p_female = float(track.p_female)
        threshold = 0.5
        if flower.sex.confidence >= 0.0:
            if track.p_female >= threshold:
                flower.sex.sex = FlowerSex.FEMALE
            else:
                flower.sex.sex = FlowerSex.MALE
        # More corroborating frames means more confidence, saturating quickly.
        flower.sex.confidence = float(np.clip(
            flower.sex.confidence * (1.0 + 0.15 * min(track.hits, 6)), 0.0, 1.0))

        flower.pollen.availability = float(track.pollen)

        dominant = track.dominant_stage()
        if dominant is not None and sum(track.stage_votes.values()) >= self.min_hits:
            flower.anthesis.stage = AnthesisStage(dominant)

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self._frame = 0
