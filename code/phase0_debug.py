import os
import numpy as np


class Phase0Logger:
    """Minimal Phase 0 logger: record every K steps and save to .npy."""
    def __init__(self, enabled: bool, out_dir: str, every: int = 20, filename: str = "debug_phase0.npy"):
        self.enabled = enabled
        self.every = every
        self.filename = filename
        self.records = []
        if self.enabled:
            os.makedirs(out_dir, exist_ok=True)
        self.out_path = os.path.join(out_dir, filename)

    @staticmethod
    def ess(w: np.ndarray) -> float:
        """Effective Sample Size (ESS) for normalized weights."""
        w = np.asarray(w, dtype=np.float64)
        s = np.sum(w)
        if s <= 0:
            return 0.0
        w = w / s
        return float(1.0 / np.sum(w * w))

    def should_record(self, time_idx: int, meas_type: str) -> bool:
        return self.enabled and (meas_type == "L") and (time_idx % self.every == 0)

    def add(self, rec: dict) -> None:
        if self.enabled:
            self.records.append(rec)

    def save(self) -> None:
        if not self.enabled:
            return
        np.save(self.out_path, np.array(self.records, dtype=object), allow_pickle=True)
