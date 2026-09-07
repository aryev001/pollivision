"""Vision-language embedding backend used for fine-grained cue scoring.

The detector answers *where is a flower*. This module answers the harder
questions the report's Sec. III-C and III-E need: is it staminate or pistillate,
is it actually receptive, and how much pollen is sitting on the anther. Those
are fine-grained distinctions that a zero-shot detector prompted at frame level
handles poorly, but that a CLIP-family model handles well on a tight crop.

Weight sourcing is the wrinkle. The model is distributed as a TorchScript bundle
that carries *both* towers' weights but only exports the text tower's forward
method, so the image tower cannot be called directly. Rather than reaching for a
model hub - which many agricultural and campus networks block - this module
lifts the state dict out of the bundle and loads it into the matching
architecture from the open-source package, recovering a fully working image
encoder from a file that is already reachable over GitHub.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np

from ..logging_utils import get_logger
from ..zoo import resolve

LOGGER = get_logger(__name__)

_ARCH_FOR_FILENAME = {
    "mobileclip_blt.ts": "mobileclip_b",
    "mobileclip_b.ts": "mobileclip_b",
    "mobileclip_s0.ts": "mobileclip_s0",
    "mobileclip_s1.ts": "mobileclip_s1",
    "mobileclip_s2.ts": "mobileclip_s2",
}


class VisionLanguageModel:
    """Zero-shot image/text scorer with a cached text-embedding table."""

    def __init__(
        self,
        weights: str = "mobileclip-blt",
        device: str = "cpu",
        temperature: float = 100.0,
        batch_size: int = 16,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.temperature = float(temperature)
        self.batch_size = int(batch_size)
        self._text_cache: dict[tuple[str, ...], "torch.Tensor"] = {}

        path = resolve(weights)
        arch = _ARCH_FOR_FILENAME.get(path.name, "mobileclip_b")
        LOGGER.info("Loading vision-language model %s (%s)", path.name, arch)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scripted = torch.jit.load(str(path), map_location=device).eval()

        import mobileclip

        model, _, preprocess = mobileclip.create_model_and_transforms(arch, pretrained=None)
        missing, unexpected = model.load_state_dict(scripted.state_dict(), strict=False)
        if missing or unexpected:
            # A partial load means the bundle and the architecture disagree; the
            # resulting embeddings would be silently meaningless.
            raise RuntimeError(
                f"Vision-language weights did not match architecture '{arch}': "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys."
            )
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        del scripted

        import clip as _clip

        self.tokenize = _clip.clip.tokenize
        self.embed_dim = int(getattr(model, "embed_dim", 512) or 512)

    # -- encoding ---------------------------------------------------------- #

    def encode_text(self, prompts: Sequence[str]):
        """Embed and L2-normalise a prompt list, memoised on the exact tuple."""
        key = tuple(prompts)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        tokens = self.tokenize(list(prompts))
        with self.torch.no_grad():
            feats = self.model.encode_text(tokens.to(self.device))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        self._text_cache[key] = feats
        return feats

    def encode_images(self, crops: Sequence[np.ndarray]):
        """Embed a list of BGR crops, returning L2-normalised features."""
        if not len(crops):
            return self.torch.zeros((0, self.embed_dim))
        from PIL import Image

        tensors = []
        for crop in crops:
            if crop is None or crop.size == 0:
                crop = np.zeros((8, 8, 3), dtype=np.uint8)
            rgb = crop[:, :, ::-1] if crop.ndim == 3 else np.stack([crop] * 3, -1)
            tensors.append(self.preprocess(Image.fromarray(np.ascontiguousarray(rgb))))

        batch = self.torch.stack(tensors).to(self.device)
        outputs = []
        with self.torch.no_grad():
            for start in range(0, len(batch), self.batch_size):
                feats = self.model.encode_image(batch[start:start + self.batch_size])
                outputs.append(feats)
        feats = self.torch.cat(outputs, dim=0)
        return feats / feats.norm(dim=-1, keepdim=True)

    # -- scoring ----------------------------------------------------------- #

    def score(self, crops: Sequence[np.ndarray], prompts: Sequence[str]) -> np.ndarray:
        """Softmax similarity of each crop against each prompt.

        Returns an ``(n_crops, n_prompts)`` array whose rows sum to 1.
        """
        if not len(crops) or not len(prompts):
            return np.zeros((len(crops), len(prompts)), dtype=np.float32)
        image_features = self.encode_images(crops)
        text_features = self.encode_text(prompts)
        logits = self.temperature * image_features @ text_features.T
        return logits.softmax(dim=-1).cpu().numpy().astype(np.float32)

    def score_groups(
        self,
        crops: Sequence[np.ndarray],
        groups: dict[str, Sequence[str]],
    ) -> list[dict[str, float]]:
        """Score crops against named groups of prompts.

        Prompts within a group are averaged before the softmax, which is the
        standard prompt-ensembling trick: it makes the result depend on the
        *concept* rather than on the phrasing of any single prompt.
        """
        names = [name for name, prompts in groups.items() if prompts]
        if not names or not len(crops):
            return [{} for _ in crops]

        flat: list[str] = []
        spans: list[tuple[int, int]] = []
        for name in names:
            prompts = list(groups[name])
            spans.append((len(flat), len(flat) + len(prompts)))
            flat.extend(prompts)

        image_features = self.encode_images(crops)
        text_features = self.encode_text(flat)

        # Average the normalised prompt embeddings per group, then renormalise.
        group_vectors = []
        for start, end in spans:
            vector = text_features[start:end].mean(dim=0)
            group_vectors.append(vector / vector.norm())
        group_matrix = self.torch.stack(group_vectors)

        logits = self.temperature * image_features @ group_matrix.T
        probabilities = logits.softmax(dim=-1).cpu().numpy().astype(np.float32)
        return [dict(zip(names, row.tolist())) for row in probabilities]


_SINGLETON: Optional[VisionLanguageModel] = None


def get_vlm(cfg, device: str = "cpu") -> Optional[VisionLanguageModel]:
    """Process-wide shared instance.

    The model is ~150 M parameters; the sex, anthesis, pollen and verification
    heads all want it, and loading four copies on a 2 GB SBC is not an option.
    """
    global _SINGLETON
    if not cfg.get("vlm.enabled", True):
        return None
    if _SINGLETON is None:
        try:
            _SINGLETON = VisionLanguageModel(
                weights=cfg.get("vlm.weights", "mobileclip-blt"),
                device=device,
                temperature=float(cfg.get("vlm.temperature", 100.0)),
                batch_size=int(cfg.get("vlm.batch_size", 16)),
            )
        except Exception as exc:  # noqa: BLE001
            # The geometric cues are designed to stand alone, so a missing VLM
            # degrades accuracy rather than stopping the rover.
            LOGGER.warning("Vision-language model unavailable (%s); "
                           "falling back to geometric cues only", exc)
            return None
    return _SINGLETON


def reset_vlm() -> None:
    """Drop the shared instance. Used by tests and by `pollivision bench`."""
    global _SINGLETON
    _SINGLETON = None
