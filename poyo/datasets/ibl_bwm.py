from collections.abc import Callable
from pathlib import Path
from typing import Literal

from torch_brain.data import Data
from torch_brain.datasets import Dataset
from torch_brain.datasets.mixins import SpikingDatasetMixin

from .wrapper import PoyoReadoutConfig


class IBLBWMDataset(SpikingDatasetMixin, Dataset):
    """IBL Brain-Wide Map sessions written by ``convert_sessions.py``.

    Decodes wheel velocity (``behavior.wheel_vel``, rad/s) from spikes.
    Each file needs ``behavior``, ``train_domain`` and ``valid_domain``
    (see ``add_wheel_targets.py``).
    """

    dim_target = 1

    def __init__(
        self,
        root: str,
        recording_ids: list[str] | None = None,
        transform: Callable | None = None,
        dirname: str = "ibl_bwm",
        normalize_std: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "units.id"],
            **kwargs,
        )
        # unit ids like "probe00_12" repeat across sessions
        self.spiking_dataset_mixin_uniquify_unit_ids = True
        self.normalize_std = normalize_std

    def get_recording_hook(self, data: Data):
        super().get_recording_hook(data)
        data.readout_config = PoyoReadoutConfig(
            value_key="behavior.wheel_vel",
            timestamp_key="behavior.timestamps",
            normalize_mean=0.0,
            normalize_std=self.normalize_std,
        )

    def get_sampling_intervals(
        self,
        split: Literal["train", "valid", "test"] | None = None,
    ):
        domain_key = "domain" if split is None else f"{split}_domain"
        return {
            rid: getattr(self.get_recording(rid), domain_key)
            for rid in self.recording_ids
        }
