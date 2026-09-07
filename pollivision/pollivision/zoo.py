"""Model weight resolution and caching.

The design constraint that shapes this module: the target user has no GPU and
may be behind a restricted network. Every weight PolliVision needs by default is
therefore (a) small enough to run on CPU, and (b) mirrored on GitHub release
assets, which are reachable from far more networks than model hubs are. Weights
resolve in this order:

1. An explicit path passed by the caller.
2. The local cache directory (so a machine can be primed once and run offline).
3. Any configured mirror URL, tried in order.

Set ``POLLIVISION_OFFLINE=1`` to forbid network access entirely and fail loudly
instead of silently stalling on a download.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .logging_utils import get_logger

LOGGER = get_logger(__name__)

_ULTRALYTICS_RELEASE = "https://github.com/ultralytics/assets/releases/download"


def cache_dir() -> Path:
    """Directory holding downloaded weights; override with ``POLLIVISION_HOME``."""
    root = os.environ.get("POLLIVISION_HOME")
    if root:
        path = Path(root).expanduser()
    else:
        path = Path.home() / ".cache" / "pollivision"
    path.mkdir(parents=True, exist_ok=True)
    return path


def offline() -> bool:
    return os.environ.get("POLLIVISION_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


@dataclass
class ModelSpec:
    """A named weight file plus where it can be fetched from."""

    name: str
    filename: str
    mirrors: list[str] = field(default_factory=list)
    sha256: Optional[str] = None
    notes: str = ""

    def cached_path(self) -> Path:
        return cache_dir() / self.filename


# Every default backend resolves through GitHub release assets, which keeps the
# stack usable on networks that block the usual model hubs.
REGISTRY: dict[str, ModelSpec] = {
    # Open-vocabulary detector + instance segmenter. Segmentation matters here:
    # the corolla mask drives orientation, occlusion and anther analysis.
    "yoloe-11s-seg": ModelSpec(
        name="yoloe-11s-seg",
        filename="yoloe-11s-seg.pt",
        mirrors=[f"{_ULTRALYTICS_RELEASE}/v8.4.0/yoloe-11s-seg.pt",
                 f"{_ULTRALYTICS_RELEASE}/v8.3.0/yoloe-11s-seg.pt"],
        notes="Primary detector: text-prompted open-vocabulary detection + masks.",
    ),
    "yoloe-11m-seg": ModelSpec(
        name="yoloe-11m-seg",
        filename="yoloe-11m-seg.pt",
        mirrors=[f"{_ULTRALYTICS_RELEASE}/v8.4.0/yoloe-11m-seg.pt",
                 f"{_ULTRALYTICS_RELEASE}/v8.3.0/yoloe-11m-seg.pt"],
        notes="Higher-accuracy detector for offline auto-labelling.",
    ),
    "yoloe-11l-seg": ModelSpec(
        name="yoloe-11l-seg",
        filename="yoloe-11l-seg.pt",
        mirrors=[f"{_ULTRALYTICS_RELEASE}/v8.4.0/yoloe-11l-seg.pt",
                 f"{_ULTRALYTICS_RELEASE}/v8.3.0/yoloe-11l-seg.pt"],
        notes="Largest open-vocab teacher; used to auto-label training data.",
    ),
    # Vision-language model used for fine-grained sex / stage / pollen scoring.
    # Shipped as a TorchScript bundle carrying BOTH towers.
    "mobileclip-blt": ModelSpec(
        name="mobileclip-blt",
        filename="mobileclip_blt.ts",
        mirrors=[f"{_ULTRALYTICS_RELEASE}/v8.4.0/mobileclip_blt.ts",
                 f"{_ULTRALYTICS_RELEASE}/v8.3.0/mobileclip_blt.ts"],
        notes="MobileCLIP-B(LT) image+text towers for zero-shot cue scoring.",
    ),
    # Monocular depth, for the ESP32-CAM configuration which has no depth channel.
    "midas-small": ModelSpec(
        name="midas-small",
        filename="midas_v21_small_256.pt",
        mirrors=["https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"],
        notes="Lightweight relative-depth network for monocular range recovery.",
    ),
    "midas-swin2-tiny": ModelSpec(
        name="midas-swin2-tiny",
        filename="dpt_swin2_tiny_256.pt",
        mirrors=["https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt"],
        notes="More accurate monocular depth; heavier on CPU.",
    ),
}


def _download(url: str, dest: Path) -> bool:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        LOGGER.info("Fetching %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "PolliVision/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response, open(tmp, "wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp.replace(dest)
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means "try the next mirror"
        LOGGER.warning("Download failed from %s: %s", url, exc)
        tmp.unlink(missing_ok=True)
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(name: str, explicit: Optional[str] = None) -> Path:
    """Return a local path to the weights for ``name``, downloading if needed."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Weights not found at explicit path: {path}")
        return path

    if name not in REGISTRY:
        # Allow a bare filename or URL that is not in the registry.
        candidate = Path(name).expanduser()
        if candidate.exists():
            return candidate
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(REGISTRY)}")

    spec = REGISTRY[name]
    dest = spec.cached_path()
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    if offline():
        raise RuntimeError(
            f"'{name}' is not cached at {dest} and POLLIVISION_OFFLINE is set. "
            f"Prime the cache on a connected machine with: pollivision fetch --model {name}"
        )

    for url in spec.mirrors:
        if _download(url, dest):
            if spec.sha256:
                actual = _sha256(dest)
                if actual != spec.sha256:
                    dest.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Checksum mismatch for {name}: expected {spec.sha256}, got {actual}"
                    )
            return dest

    raise RuntimeError(
        f"Could not fetch '{name}' from any mirror ({spec.mirrors}). "
        "If your network blocks these hosts, download the file elsewhere and place it at "
        f"{dest}."
    )


def fetch_all(names: Optional[list[str]] = None) -> dict[str, Path]:
    """Prime the cache. Used by ``pollivision fetch`` before going to the field."""
    names = names or ["yoloe-11s-seg", "mobileclip-blt"]
    return {name: resolve(name) for name in names}


def configure_ultralytics_cache() -> None:
    """Point Ultralytics' own asset downloader at the PolliVision cache.

    Ultralytics resolves auxiliary assets (notably the 572 MB MobileCLIP text
    encoder that open-vocabulary prompting needs) through its own downloader,
    which checks the current working directory and then its configured weights
    directory. Left alone it scatters large files into whatever directory the
    rover happens to be launched from. Redirecting it here keeps every weight in
    one cache that can be primed once and reused offline.
    """
    try:
        from ultralytics.utils import SETTINGS

        target = str(cache_dir())
        if SETTINGS.get("weights_dir") != target:
            SETTINGS["weights_dir"] = target
    except Exception as exc:  # noqa: BLE001 - never fatal, only tidiness
        LOGGER.debug("Could not redirect Ultralytics weights_dir: %s", exc)
