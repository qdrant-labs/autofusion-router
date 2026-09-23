# Autofusion Router

Routes a search query to dense, sparse, or RRF retrieval from word rarity alone. It reads a query's
wordfreq zipf scores into a small MLP and returns a route, or a fusion weight for the sparse leg.
Neither port needs torch or a GPU.

| port | path | runtime deps |
|---|---|---|
| Python | [`python/`](python/README.md) | numpy, wordfreq |
| TypeScript | [`js/`](js/README.md) | none |

```python
from autofusion_router import Router

Router().classify("CVE-2021-44228 log4j")   # [Classification(query, route, p_dense, p_sparse)]
```

Each port's README covers its API, install and dev setup, the licence levels, and which arm a build ships.

## Layout

```
bundle/       the data both ports load; generated, don't edit
  arms.json     router weights for the shipped arm (float32, base64)
  zipf_en.json  English word-rarity table derived from wordfreq (CC-BY-SA-4.0)
  golden.json   reference outputs from the torch serving path, the ports' test oracle
python/       Python package (hatchling)
js/           TypeScript package, zero dependencies
```

## Regenerating the bundle

This repo can't regenerate `bundle/`. The exporter needs the trained arms and the training data,
and both live in the parent research repo. From the parent's root:

```bash
poetry run python src/scripts/export_autofusion_bundle.py --allow-license PERMISSIVE
```

That rewrites `autofusion_router/bundle/`. Commit here, then commit the submodule pointer in the
parent. Both ports' test suites (`pytest` in `python/`, `npm test` in `js/`) check every English
query in `golden.json`, so run them after each export.

## Licence

Code is Apache-2.0 (`LICENSE`). `bundle/zipf_en.json` is CC-BY-SA-4.0, and the query strings in
`bundle/golden.json` come from the collections listed in `NOTICE`.
