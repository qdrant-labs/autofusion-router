# @qdrant-labs/autofusion-router

Routes a search query to dense, sparse, or RRF retrieval from word rarity alone. It's a
zero-dependency TypeScript port of the router service's zipf-only arms, and runs in Node ≥ 20.10,
Bun, Deno, browsers, and edge runtimes.

```ts
import { Arm, Router } from "@qdrant-labs/autofusion-router";

const router = new Router();                          // the one arm this build ships
router.current;                                       // "v3_zipf_only_permissive_sharealike"
router.getArms();                                     // [{ name, trained_license, trained_at }]
router.classify("CVE-2021-44228 log4j");              // [{ query, route, p_dense, p_sparse }]
router.classifyPure(["why do cats purr"]);            // ["sparse_only"]
router.weighted("oil rubbed bronze door knob");       // [{ query, alpha, p_dense, p_sparse }]
```

`weighted()` returns the TM2C2 weight on the sparse leg, so you fuse
`alpha * sparse + (1 - alpha) * dense` over top-normalized scores. It isn't an RRF weight.

Queries are read as English. The Python service reads other languages in their own wordfreq
table, so non-English routes can differ.

## Which arm ships

Every build ships one arm: the same zipf-only model, trained on the collections the build's
licence level allows. The maintainer picks the level at build time, not the client;
`export_autofusion_bundle.py --allow-license` (in the parent repo) uses the router service's `ALLOW_LICENSE` levels:

| level | arm | trained on |
|---|---|---|
| `NONE` | `v3_zipf_only_permissive` | permissive (CC-BY / Apache / MIT) collections |
| `PERMISSIVE` | `v3_zipf_only_permissive_sharealike` | those plus CC-BY-SA collections |
| `ALL` | `v3_zipf_only` | every collection, including non-commercial ones such as MS MARCO |

`router.allowLicense` reports the level the build was exported at, and the arm's `trained_lanes`
in `data/arms.json` lists its collections. See `NOTICE` for attributions and the word-table licence.

## Development

The data comes from `../bundle`, which the parent repo's `src/scripts/export_autofusion_bundle.py --allow-license <level>` writes from
the trained arms.
`npm test` builds the package and checks every English golden query against the torch serving path.

## Publishing (GitHub Packages, private)

The package is published to GitHub Packages, so only people with read access to
`qdrant-labs/search-routing-experiments` can install it. `publishConfig` keeps `npm publish` off
the public registry.

```bash
poetry run python src/scripts/export_autofusion_bundle.py --allow-license PERMISSIVE   # what ships, from the parent repo
npm login --scope=@qdrant-labs --auth-type=legacy --registry=https://npm.pkg.github.com
cd autofusion_router/js && npm pack --dry-run && npm publish            # prepublishOnly runs the tests
```

To install, add a project `.npmrc` with a GitHub token that has `read:packages`:

```
@qdrant-labs:registry=https://npm.pkg.github.com
//npm.pkg.github.com/:_authToken=${GITHUB_TOKEN}
```
