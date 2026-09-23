// The port against golden.json — outputs of the torch serving path (export_autofusion_bundle.py in the parent repo) on
// permissive-lane queries. English rows only: the port reads every query as English.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { Arm, Route, Router, zipf, zipfFeatures } from "../dist/index.js";

const golden = JSON.parse(readFileSync(new URL("../../bundle/golden.json", import.meta.url)));
const table = JSON.parse(readFileSync(new URL("../data/zipf_en.json", import.meta.url)));
const english = golden.queries.map((q, i) => [q, i]).filter(([q]) => q.lang === "en");
const EPS = 1e-5; // torch runs float32, the port float64 over the same float32 weights
const shipped = JSON.parse(readFileSync(new URL("../data/arms.json", import.meta.url)));

test("every golden token scores exactly as wordfreq", () => {
  const wrong = Object.entries(golden.tokens).filter(([t, z]) => zipf(t, table) !== z);
  assert.deepEqual(wrong.map(([t, z]) => [t, z, zipf(t, table)]), []);
});

test("features match the Python path", () => {
  for (const [q] of english) {
    zipfFeatures(q.query, table, 3.0).forEach((f, j) =>
      assert.ok(Math.abs(f - q.features[j]) < 1e-6, `${JSON.stringify(q.query)} feature ${j}: ${f} vs ${q.features[j]}`));
  }
});

for (const [name, expected] of Object.entries(golden.arms)) {
  test(`${name}: probabilities, routes and alpha match torch`, () => {
    const router = new Router({ arm: name });
    assert.equal(router.current, name);
    const [tS, tD] = shipped.arms[name].thresholds;
    let near = 0;
    for (const [q, i] of english) {
      const want = expected.results[i];
      const [got] = router.classify(q.query);
      const [w] = router.weighted(q.query);
      assert.ok(Math.abs(got.p_dense - want.p_dense) < EPS && Math.abs(got.p_sparse - want.p_sparse) < EPS,
        `${JSON.stringify(q.query)}: p ${got.p_sparse},${got.p_dense} vs ${want.p_sparse},${want.p_dense}`);
      // a route may flip only where a probability sits within float32 noise of a decision boundary
      const edge = Math.abs(Math.abs(want.p_dense - want.p_sparse) - expected.delta) < EPS
        || Math.abs(want.p_sparse - tS) < EPS || Math.abs(want.p_dense - tD) < EPS;
      if (edge) { near++; continue; }
      assert.equal(got.route, want.route, JSON.stringify(q.query));
      assert.ok(Math.abs(w.alpha - want.alpha) < 1e-4, `${JSON.stringify(q.query)}: alpha ${w.alpha} vs ${want.alpha}`);
    }
    assert.ok(near <= 2, `${near} rows on a decision boundary`);
  });
}

test("starts on the maintainer's default and chooses only shipped arms", () => {
  const router = new Router();
  assert.equal(router.current, shipped.default);
  assert.equal(router.allowLicense, shipped.allow_license);
  assert.deepEqual(router.getArms().map((a) => a.name), Object.keys(shipped.arms));
  assert.deepEqual(Object.values(Arm), Object.keys(shipped.arms));   // the constants name exactly the shipped arms
  for (const arm of Object.values(Arm)) assert.equal(new Router({ arm }).current, arm);
  assert.equal(router.choose(Object.values(Arm)[0]).current, Object.values(Arm)[0]);
  router.choose(shipped.default);
  assert.deepEqual(Object.keys(golden.arms).sort(), Object.keys(shipped.arms).sort());
  assert.throws(() => router.choose("not_shipped"), /this build ships/);
  assert.throws(() => new Router({ arm: "not_shipped" }), /this build ships/);
  assert.equal(router.current, shipped.default);
  const queries = ["CVE-2021-44228 log4j remote code execution", "why do cats purr"];
  assert.deepEqual(router.classifyPure(queries), router.classify(queries).map((c) => c.route));
  assert.equal(router.classify("why do cats purr").length, 1);
  const routes = Object.values(golden.arms).flatMap((a) => a.results.map((r) => r.route));
  assert.ok(routes.every((r) => Object.values(Route).includes(r)));   // Route names every route torch emits
});
