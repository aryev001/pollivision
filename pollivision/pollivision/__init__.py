"""PolliVision - perception and adaptive-control stack for a vision-guided
electrostatic pollination rover.

Implements the vision, localisation and control pipeline described in
"Adaptive Vision-Guided Electrostatic Pollination Rover for Precision
Pollination of Ground-Level Cucurbit Crops": flower detection and sex
classification (Sec. III-C), approach localisation (Sec. III-D), adaptive
electrostatic parameter selection (Sec. III-E/F), and outcome verification
closing the loop of Fig. 1 (Sec. III-G).

The stack runs inference-only on a CPU. Nothing in the default path requires
training or a GPU.
"""

__version__ = "1.0.0"

from .config import Config, load_config  # noqa: F401
from .types import (  # noqa: F401
    AnthesisStage,
    ElectrostaticCommand,
    EnvironmentReading,
    FlowerObservation,
    FlowerSex,
    FrameResult,
    ProbeMode,
)

__all__ = [
    "Config",
    "load_config",
    "AnthesisStage",
    "ElectrostaticCommand",
    "EnvironmentReading",
    "FlowerObservation",
    "FlowerSex",
    "FrameResult",
    "ProbeMode",
    "__version__",
]
