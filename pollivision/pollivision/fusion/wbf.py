"""Weighted Boxes Fusion across detector backends.

Non-maximum suppression picks one winner per cluster and throws the rest away.
That is the wrong behaviour when merging heterogeneous detectors: an
open-vocabulary model and a supervised model that both localise the same flower
slightly differently carry complementary information, and averaging their boxes
is measurably more accurate than picking either. WBF averages the cluster
instead, weighting each member by (backend trust x detection score).

Masks are carried through from the highest-scoring mask-bearing member of each
cluster rather than averaged - a soft-averaged mask has fuzzy borders, and every
downstream consumer here (ellipse fit, occlusion, anther extraction) wants a
crisp instance boundary.
"""

from __future__ import annotations

import numpy as np

from ..types import BBox, Detection


def fuse(
    detections: list[Detection],
    iou_threshold: float = 0.55,
    score_threshold: float = 0.0,
    require_votes: int = 1,
    n_backends: int = 1,
) -> list[Detection]:
    """Fuse detections from one or more backends.

    Args:
        detections: Candidates from every backend, already role-labelled.
        iou_threshold: Minimum IoU for a detection to join an existing cluster.
        score_threshold: Drop fused boxes scoring below this.
        require_votes: Number of *distinct backends* that must contribute to a
            cluster for it to survive. Raising this to 2 trades recall for
            precision and only makes sense with 2+ backends enabled.
        n_backends: How many backends are actually running, used to normalise
            the fused score.

    Returns:
        Fused detections, sorted by descending score.
    """
    if not detections:
        return []

    # Fuse within a role: a flower and an ovary overlap heavily by construction
    # (the ovary sits directly under the corolla) and must not be merged.
    by_label: dict[str, list[Detection]] = {}
    for det in detections:
        by_label.setdefault(det.label, []).append(det)

    fused: list[Detection] = []
    for label, group in by_label.items():
        fused.extend(
            _fuse_single_label(
                group, label, iou_threshold, score_threshold, require_votes, n_backends
            )
        )
    fused.sort(key=lambda d: d.score, reverse=True)
    return fused


def _fuse_single_label(
    detections: list[Detection],
    label: str,
    iou_threshold: float,
    score_threshold: float,
    require_votes: int,
    n_backends: int,
) -> list[Detection]:
    ordered = sorted(detections, key=lambda d: d.score * _trust(d), reverse=True)

    clusters: list[list[Detection]] = []
    cluster_boxes: list[BBox] = []

    for det in ordered:
        best_index, best_iou = -1, 0.0
        for index, box in enumerate(cluster_boxes):
            value = det.box.iou(box)
            if value > best_iou:
                best_index, best_iou = index, value
        if best_index >= 0 and best_iou >= iou_threshold:
            clusters[best_index].append(det)
            cluster_boxes[best_index] = _weighted_box(clusters[best_index])
        else:
            clusters.append([det])
            cluster_boxes.append(det.box)

    results: list[Detection] = []
    for members, box in zip(clusters, cluster_boxes):
        sources = sorted({m.source for m in members})
        if len(sources) < require_votes:
            continue

        weights = np.array([m.score * _trust(m) for m in members], dtype=np.float64)
        total = float(weights.sum())
        if total <= 1e-9:
            continue

        # Confidence-weighted mean score, then scaled by how much of the
        # available backend agreement this cluster actually attracted. A box
        # found by one of three backends should not score like a unanimous one.
        mean_score = float((np.array([m.score for m in members]) * weights).sum() / total)
        agreement = min(len(sources), max(n_backends, 1)) / max(n_backends, 1)
        score = float(np.clip(mean_score * (0.5 + 0.5 * agreement), 0.0, 1.0))
        if score < score_threshold:
            continue

        mask = None
        for member in sorted(members, key=lambda m: m.score, reverse=True):
            if member.mask is not None:
                mask = member.mask
                break

        results.append(
            Detection(
                box=box,
                score=score,
                label=label,
                source="+".join(sources),
                mask=mask,
                meta={
                    "n_members": len(members),
                    "sources": sources,
                    "member_scores": [round(float(m.score), 4) for m in members],
                    "prompts": sorted({str(m.meta.get("prompt", "")) for m in members} - {""}),
                },
            )
        )
    return results


def _trust(det: Detection) -> float:
    return float(det.meta.get("backend_weight", 1.0))


def _weighted_box(members: list[Detection]) -> BBox:
    """Score-weighted average of a cluster's corner coordinates."""
    weights = np.array([m.score * _trust(m) for m in members], dtype=np.float64)
    if weights.sum() <= 1e-9:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    corners = np.stack([m.box.as_array() for m in members]).astype(np.float64)
    averaged = (corners * weights[:, None]).sum(axis=0)
    return BBox.from_xyxy(averaged)
