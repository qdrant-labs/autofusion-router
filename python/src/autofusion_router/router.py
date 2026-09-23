"""Zipf-only router arms from the exported bundle, served with numpy and wordfreq: the
router_service serving path for these arms, without torch. English only."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from importlib.resources import files
from typing import Literal

import numpy as np
from wordfreq import zipf_frequency

TrainedLicense = Literal["all", "permissive_sharealike", "permissive"]
_WORD = re.compile(r"[a-z0-9]+")
_ACTIVATIONS = {
    "relu": lambda x: np.maximum(x, 0.0),
    "none": lambda x: x,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
}


def zipf_features(queries: list[str], rare: float) -> np.ndarray:
    """(rows, 5) float32: min, mean, max, rare share, OOV share of each query's token rarities."""
    rows = np.zeros((len(queries), 5), dtype=np.float32)
    for i, text in enumerate(queries):
        tokens = _WORD.findall(text.lower())
        if tokens:
            freqs = np.array([zipf_frequency(t, "en") for t in tokens])
            rows[i] = freqs.min(), freqs.mean(), freqs.max(), (freqs < rare).mean(), (freqs == 0.0).mean()
    return rows


def _float32s(encoded: str, shape: list[int]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(encoded), dtype="<f4").reshape(shape)


@cache
def manifest() -> dict:
    """What the maintainer shipped: the arms its ALLOW_LICENSE level permits, and the default."""
    return json.loads((files(__package__) / "arms.json").read_text())


Arm = StrEnum("Arm", {name.upper(): name for name in manifest()["arms"]})
"""The arms this build ships, e.g. `Arm.V3_ZIPF_ONLY_PERMISSIVE`; plain strings work too."""


@dataclass(frozen=True)
class Classification:
    query: str
    route: str
    p_dense: float
    p_sparse: float


@dataclass(frozen=True)
class Weighted:
    query: str
    alpha: float
    """TM2C2 weight on the sparse leg: fuse alpha * sparse + (1 - alpha) * dense."""
    p_dense: float
    p_sparse: float


@dataclass(frozen=True)
class ArmInfo:
    name: str
    trained_license: str
    trained_at: str


class ArmModel:
    """One arm: query text -> route, head probabilities, or TM2C2 fusion weight."""

    def __init__(self, bundle: dict) -> None:
        self.bundle = bundle
        self.arm: str = bundle["arm"]
        self.trained_license: str = bundle["trained_license"]
        self.trained_at: str = bundle["trained_at"]
        self.delta: float = bundle["rrf_delta"]
        self.k: float = bundle["fusion_k"]
        heads = bundle["head_routes"]
        self._sparse, self._dense = heads.index("sparse_only"), heads.index("dense_only")
        self._thresholds = np.asarray(bundle["thresholds"], np.float32)
        self._mean = np.asarray(bundle["zipf_mean"], np.float32)
        self._std = np.asarray(bundle["zipf_std"], np.float32)
        self._layers = [
            (_float32s(layer["weight"], layer["shape"]), _float32s(layer["bias"], layer["shape"][:1]),
             _ACTIVATIONS[layer["activation"]])
            for layer in bundle["layers"]
        ]

    def probabilities(self, queries: list[str]) -> np.ndarray:
        """(rows, heads) in the bundle's `head_routes` order."""
        x = (zipf_features(queries, self.bundle["zipf_rare"]) - self._mean) / self._std
        for weight, bias, act in self._layers:
            x = act(x @ weight.T + bias)
        return x

    def _serve(self, probs: np.ndarray) -> np.ndarray:
        """Most probable head among those clearing their threshold, else the most probable head."""
        choices = np.array([*self.bundle["head_routes"], self.bundle["hedge"]], dtype=object)
        fires = probs >= self._thresholds
        fired = np.where(fires, probs, -np.inf).argmax(axis=1)
        doubt = probs.max(axis=1) < self.bundle["hedge_bar"]
        fallback = np.where(doubt, probs.shape[1], probs.argmax(axis=1))
        return choices[np.where(fires.any(axis=1), fired, fallback)]

    def classify(self, queries: str | list[str], delta: float | None = None) -> list[Classification]:
        qs = [queries] if isinstance(queries, str) else list(queries)
        if not qs:
            return []
        probs = self.probabilities(qs)
        s, d = probs[:, self._sparse], probs[:, self._dense]
        delta = self.delta if delta is None else delta
        route = np.where(np.abs(d - s) < delta, self.bundle["hedge"], self._serve(probs))
        return [Classification(q, str(route[i]), float(d[i]), float(s[i])) for i, q in enumerate(qs)]

    def classify_pure(self, queries: str | list[str], delta: float | None = None) -> list[str]:
        return [r.route for r in self.classify(queries, delta)]

    def weighted(
        self, queries: str | list[str], k: float | None = None, delta: float | None = None
    ) -> list[Weighted]:
        """TM2C2 weight on the sparse leg: each head's probability re-centred on its own
        threshold, scaled by k; fuse alpha * sparse + (1 - alpha) * dense."""
        qs = [queries] if isinstance(queries, str) else list(queries)
        if not qs:
            return []
        probs = self.probabilities(qs)
        s, d = probs[:, self._sparse], probs[:, self._dense]
        t_s, t_d = self._thresholds[self._sparse], self._thresholds[self._dense]
        k = self.k if k is None else k
        delta = self.delta if delta is None else delta
        fires_s, fires_d = s >= t_s, d >= t_d
        sparse_wins = np.where(fires_s ^ fires_d, fires_s, s >= d)
        prob, bar = np.where(sparse_wins, s, d), np.where(sparse_wins, t_s, t_d)
        lean = np.clip(0.5 + 0.5 * (prob - bar) / (1.0 - bar), 0.0, 1.0)
        anchored = np.where(sparse_wins, lean, 1.0 - lean)
        alpha = np.clip(0.5 + k * (anchored - 0.5), 0.0, 1.0)
        alpha = np.where(np.abs(d - s) < delta, 0.5, alpha)
        return [Weighted(q, float(alpha[i]), float(d[i]), float(s[i])) for i, q in enumerate(qs)]


class Router:
    """Routes queries with one of the shipped arms; without `arm`, the maintainer's default."""

    def __init__(self, arm: Arm | str | None = None) -> None:
        self._loaded: dict[str, ArmModel] = {}
        self._arm = self._load(manifest()["default"] if arm is None else arm)

    def _load(self, name: str) -> ArmModel:
        arms = manifest()["arms"]
        if name not in arms:
            raise ValueError(f"no arm {name!r}; this build ships {', '.join(arms)}")
        if name not in self._loaded:
            self._loaded[name] = ArmModel(arms[name])
        return self._loaded[name]

    def get_arms(self) -> list[ArmInfo]:
        return [ArmInfo(a["arm"], a["trained_license"], a["trained_at"]) for a in manifest()["arms"].values()]

    @property
    def current(self) -> Arm:
        return Arm(self._arm.arm)

    @property
    def allow_license(self) -> str:
        """The licence level this build was exported at."""
        return manifest()["allow_license"]

    def choose(self, name: Arm | str) -> Router:
        self._arm = self._load(name)
        return self

    def classify(self, queries: str | list[str], delta: float | None = None) -> list[Classification]:
        return self._arm.classify(queries, delta)

    def classify_pure(self, queries: str | list[str], delta: float | None = None) -> list[str]:
        return self._arm.classify_pure(queries, delta)

    def weighted(
        self, queries: str | list[str], k: float | None = None, delta: float | None = None
    ) -> list[Weighted]:
        return self._arm.weighted(queries, k, delta)
