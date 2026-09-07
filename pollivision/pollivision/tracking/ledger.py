"""Pollination ledger: what has already been serviced.

Without this the rover happily re-pollinates the same flower every time it
re-enters the frame, which wastes pollen, wastes time, and risks damaging a
pistil through repeated exposure. The ledger records outcomes per track and
enforces a cooldown and an attempt limit.

Identity here is deliberately conservative: a track ID is only meaningful for as
long as the track lives, so entries also carry the last known rover-frame
position. A flower re-acquired after the track was dropped is matched back to
its ledger entry by position, which is what makes the memory survive the rover
looking away and back again.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from ..types import FlowerSex

LOGGER = get_logger(__name__)


@dataclass
class LedgerEntry:
    """Record of the rover's dealings with one flower."""

    track_id: int
    sex: str = FlowerSex.UNKNOWN.value
    attempts: int = 0
    successes: int = 0
    last_attempt_ts: float = 0.0
    last_position: Optional[list[float]] = None
    notes: list[str] = field(default_factory=list)

    @property
    def serviced(self) -> bool:
        return self.successes > 0


class PollinationLedger:
    """Tracks which flowers have been pollinated and which to skip."""

    #: A re-acquired flower within this distance of a ledger entry is treated as
    #: the same flower. Comfortably larger than localisation noise, comfortably
    #: smaller than the spacing between neighbouring blooms.
    REACQUIRE_RADIUS_M = 0.06

    def __init__(self, cfg) -> None:
        section = cfg.section("ledger")
        self.enabled = bool(section.get("enabled", True))
        self.cooldown_s = float(section.get("revisit_cooldown_s", 900))
        self.max_attempts = int(section.get("max_attempts", 3))
        self.persist_path = section.get("persist_path")

        self.entries: dict[int, LedgerEntry] = {}
        self._pollen_source: Optional[int] = None
        self._pollen_charge: float = 0.0

        if self.persist_path:
            self._load()

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def should_skip(self, track_id: Optional[int],
                    position: Optional[np.ndarray] = None) -> tuple[bool, str]:
        """Whether this flower should be passed over, and why."""
        if not self.enabled or track_id is None:
            return False, ""

        entry = self.entries.get(track_id) or self._match_by_position(position)
        if entry is None:
            return False, ""

        if entry.attempts >= self.max_attempts and not entry.serviced:
            return True, f"exhausted {entry.attempts} attempts"

        if entry.serviced:
            elapsed = time.time() - entry.last_attempt_ts
            if elapsed < self.cooldown_s:
                return True, f"pollinated {int(elapsed)}s ago"
        return False, ""

    def _match_by_position(self, position: Optional[np.ndarray]) -> Optional[LedgerEntry]:
        """Recover a ledger entry for a flower whose track was lost and remade."""
        if position is None:
            return None
        point = np.asarray(position, dtype=np.float64).reshape(3)
        best, best_distance = None, self.REACQUIRE_RADIUS_M
        for entry in self.entries.values():
            if entry.last_position is None:
                continue
            distance = float(np.linalg.norm(point - np.asarray(entry.last_position)))
            if distance < best_distance:
                best, best_distance = entry, distance
        return best

    # ------------------------------------------------------------------ #
    # Updates
    # ------------------------------------------------------------------ #

    def record_attempt(self, track_id: int, sex: FlowerSex,
                       position: Optional[np.ndarray] = None) -> LedgerEntry:
        entry = self.entries.setdefault(track_id, LedgerEntry(track_id=track_id))
        entry.sex = sex.value
        entry.attempts += 1
        entry.last_attempt_ts = time.time()
        if position is not None:
            entry.last_position = [float(v) for v in np.asarray(position).reshape(3)]
        return entry

    def record_outcome(self, track_id: int, success: bool, note: str = "") -> None:
        entry = self.entries.get(track_id)
        if entry is None:
            return
        if success:
            entry.successes += 1
        if note:
            entry.notes.append(note)
            entry.notes = entry.notes[-5:]
        if self.persist_path:
            self._save()

    # ------------------------------------------------------------------ #
    # Probe pollen state
    # ------------------------------------------------------------------ #

    @property
    def probe_charge(self) -> float:
        """How much pollen the probe currently carries, in the 0-1 units the
        pollen estimator uses. Drives the collect-versus-deposit decision."""
        return self._pollen_charge

    @property
    def pollen_source(self) -> Optional[int]:
        """Track ID of the male flower the current pollen came from."""
        return self._pollen_source

    def load_probe(self, track_id: int, amount: float) -> None:
        self._pollen_source = track_id
        self._pollen_charge = float(np.clip(amount, 0.0, 1.0))
        LOGGER.debug("Probe loaded from track %s: charge %.2f", track_id, self._pollen_charge)

    def consume_probe(self, fraction: float) -> None:
        self._pollen_charge = float(np.clip(self._pollen_charge - fraction, 0.0, 1.0))
        if self._pollen_charge <= 1e-3:
            self._pollen_source = None

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        return {
            "tracked": len(self.entries),
            "attempts": sum(e.attempts for e in self.entries.values()),
            "successes": sum(e.successes for e in self.entries.values()),
            "females_serviced": sum(
                1 for e in self.entries.values()
                if e.serviced and e.sex == FlowerSex.FEMALE.value
            ),
            "probe_charge": round(self._pollen_charge, 3),
        }

    def _save(self) -> None:
        try:
            path = Path(self.persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"entries": [asdict(e) for e in self.entries.values()]}
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            LOGGER.warning("Could not persist ledger: %s", exc)

    def _load(self) -> None:
        try:
            path = Path(self.persist_path)
            if not path.exists():
                return
            payload = json.loads(path.read_text(encoding="utf-8"))
            for record in payload.get("entries", []):
                entry = LedgerEntry(**record)
                self.entries[entry.track_id] = entry
            LOGGER.info("Restored %d ledger entries", len(self.entries))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not load ledger: %s", exc)
