"""Route a search query to dense, sparse, or RRF retrieval from word rarity alone."""

from autofusion_router.router import Arm, ArmInfo, Classification, Route, Router, TrainedLicense, Weighted, zipf_features

__all__ = ["Arm", "ArmInfo", "Classification", "Route", "Router", "TrainedLicense", "Weighted", "zipf_features"]
