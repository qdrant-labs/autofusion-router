"""The package against golden.json, the torch serving path's outputs (export_autofusion_bundle.py in the parent repo), English rows."""

import json
from pathlib import Path

import numpy as np
import pytest

from autofusion_router import Arm, Route, Router, zipf_features
from autofusion_router.router import manifest

GOLDEN = json.loads((Path(__file__).parents[2] / "bundle" / "golden.json").read_text())
ENGLISH = [(i, q) for i, q in enumerate(GOLDEN["queries"]) if q["lang"] == "en"]
EPS = 1e-5


def test_features_match_the_python_path():
    got = zipf_features([q["query"] for _, q in ENGLISH], rare=3.0)
    np.testing.assert_allclose(got, [q["features"] for _, q in ENGLISH], atol=1e-6)


@pytest.mark.parametrize("arm", list(Arm))
def test_probabilities_routes_and_alpha_match_torch(arm):
    router = Router(arm)
    want = GOLDEN["arms"][arm]
    texts = [q["query"] for _, q in ENGLISH]
    rows = [want["results"][i] for i, _ in ENGLISH]
    for got, w, row in zip(router.classify(texts), router.weighted(texts), rows, strict=True):
        assert abs(got.p_dense - row["p_dense"]) < EPS and abs(got.p_sparse - row["p_sparse"]) < EPS
        assert got.route is Route(row["route"]), got.query
        assert abs(w.alpha - row["alpha"]) < 1e-4, got.query


def test_starts_on_the_maintainers_default_and_chooses_only_shipped_arms():
    router = Router()
    assert router.current == manifest()["default"]
    assert router.allow_license == manifest()["allow_license"]
    assert sorted(a.name for a in router.get_arms()) == sorted(GOLDEN["arms"]) == sorted(Arm)
    assert all(Router(arm).current is arm for arm in Arm)
    assert Router(str(next(iter(Arm)))).current is next(iter(Arm))   # a plain string works too
    with pytest.raises(ValueError, match="this build ships"):
        router.choose("not_shipped")
    with pytest.raises(ValueError, match="this build ships"):
        Router("not_shipped")
    assert router.current == manifest()["default"]
    assert router.classify_pure("why do cats purr") == [router.classify("why do cats purr")[0].route]
    assert router.classify([]) == []
