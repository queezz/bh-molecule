from dataclasses import dataclass
import numpy as np
from pathlib import Path
from .physics import BHModel
from .dataio import hash_params, save_npz, load_npz

@dataclass
class FitResult:
    params: dict
    y_fit: np.ndarray
    meta: dict

class FrameDataset:
    def __init__(self, image_stack: np.ndarray, wl_axis_nm: np.ndarray, channels: list[int]):
        self.stack = image_stack       # shape: (frames, channels, pixels)
        self.wl = wl_axis_nm
        self.channels = channels

    def frames(self):
        for i in range(self.stack.shape[0]):
            yield i, self.stack[i]

class BHFitter:
    def __init__(self, model: BHModel, cache_dir: Path | None = None):
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir: self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, frame_id: int, chan: int, params: dict) -> Path | None:
        if not self.cache_dir: return None
        key = hash_params({"frame": frame_id, "chan": chan, **params})
        return self.cache_dir / f"fit_{key}.npz"

    def fit_frame_channel(self, x_nm: np.ndarray, y: np.ndarray, init_params: dict, bounds=None) -> FitResult:
        # Placeholder for optimizer (curve_fit/lmfit). For now, just evaluate the forward model.
        y_fit = self.model.full_fit_model(x_nm, **init_params)
        return FitResult(params=init_params, y_fit=y_fit, meta={"ok": True})

    def fit_dataset(self, ds: FrameDataset, shared_params: dict, per_channel_params: dict[int, dict]):
        results = {}
        for frame_id, frame in ds.frames():
            for chan in ds.channels:
                y = frame[chan]
                params = {**shared_params, **per_channel_params.get(chan, {})}
                cpath = self._cache_key(frame_id, chan, params)
                if cpath and cpath.exists():
                    z = load_npz(cpath)
                    results[(frame_id, chan)] = FitResult(params=params, y_fit=z["y_fit"], meta={"cached": True})
                    continue
                out = self.fit_frame_channel(ds.wl, y, params)
                results[(frame_id, chan)] = out
                if cpath: save_npz(cpath, y_fit=out.y_fit)
        return results
