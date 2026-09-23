# autofusion-router

Routes a search query to dense, sparse, or RRF retrieval from word rarity alone. It's the router
service's zipf-only arms, served with numpy and wordfreq, and needs no torch.

It isn't on PyPI. Install a release tag from the repo, which needs read access to
`qdrant-labs/autofusion-router`:

    pip install "git+https://github.com/qdrant-labs/autofusion-router.git@python-v1.0.0#subdirectory=python"

```python
from autofusion_router import Arm, Router

router = Router()                                  # the one arm this build ships
router.current                                     # Arm.V3_ZIPF_ONLY_PERMISSIVE_SHAREALIKE
router.get_arms()                                  # [ArmInfo(name, trained_license, trained_at)]
router.classify("CVE-2021-44228 log4j")            # [Classification(query, route, p_dense, p_sparse)]
router.classify_pure(["why do cats purr"])         # ["sparse_only"]
router.weighted("oil rubbed bronze door knob")     # [Weighted(query, alpha, p_dense, p_sparse)]
```

`weighted()` returns the TM2C2 weight on the sparse leg, so you fuse
`alpha * sparse + (1 - alpha) * dense` over top-normalized scores. It isn't an RRF weight.
Queries are read as English.

Every build ships one arm: the same zipf-only model, trained on the collections the build's
licence level allows. The maintainer picks the level at build time with
the parent repo's `export_autofusion_bundle.py --allow-license NONE | PERMISSIVE | ALL` (the router service's `ALLOW_LICENSE`
levels); `router.allow_license` reports it.

| level | arm | trained on |
|---|---|---|
| `NONE` | `v3_zipf_only_permissive` | permissive (CC-BY / Apache / MIT) collections |
| `PERMISSIVE` | `v3_zipf_only_permissive_sharealike` | those plus CC-BY-SA collections |
| `ALL` | `v3_zipf_only` | every collection, including non-commercial ones such as MS MARCO |

The arm's `trained_lanes` in `arms.json` lists its collections; see `NOTICE`.

## Development

`arms.json` comes from `../bundle`, written by the parent repo's `src/scripts/export_autofusion_bundle.py --allow-license <level>`. `src/autofusion_router/arms.json`
is a committed symlink to it; the sdist replaces the link with the real file, so it builds on its own.

    pip install -e '.[test]' && pytest

On Windows, git checks the symlink out as a 25-byte text file unless symlinks are enabled, and
`import autofusion_router` then fails with `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`.
A wheel built straight from such a clone ships that text file too, so build releases with `uv build`,
which goes through the sdist. To fix the clone, turn on Developer Mode, then:

    git config core.symlinks true
    git checkout -- src/autofusion_router/arms.json
