// autofusion-router: routes a search query to dense, sparse, or RRF search from word rarity
// alone — the zipf-only arms of router_service, ported from the exported bundle.

import shipped from "../data/arms.json" with { type: "json" };
import zipfTable from "../data/zipf_en.json" with { type: "json" };

import { Arm, type ArmName } from "./arms.js";
import { type ZipfTable, zipfFeatures } from "./zipf.js";

export { Arm, type ArmName } from "./arms.js";
export { zipf, zipfFeatures } from "./zipf.js";

export type TrainedLicense = "all" | "permissive_sharealike" | "permissive";
export type Route = "dense_only" | "sparse_only" | "pure_rrf";

export interface Classification {
  query: string;
  route: Route;
  p_dense: number;
  p_sparse: number;
}

export interface Weighted {
  query: string;
  /** TM2C2 weight on the sparse leg: fuse alpha * sparse + (1 - alpha) * dense over top-normalized scores. */
  alpha: number;
  p_dense: number;
  p_sparse: number;
}

interface Layer {
  shape: [number, number];
  activation: "relu" | "none" | "sigmoid";
  weight: string;
  bias: string;
}

export interface ArmBundle {
  arm: string;
  trained_license: TrainedLicense;
  trained_at: string;
  rrf_delta: number;
  fusion_k: number;
  head_routes: Route[];
  hedge: Route;
  hedge_bar: number;
  thresholds: number[];
  zipf_rare: number;
  zipf_mean: number[];
  zipf_std: number[];
  layers: Layer[];
}

/** What the maintainer shipped: the arms its ALLOW_LICENSE level permits, and the default among them. */
interface Manifest {
  allow_license: "ALL" | "PERMISSIVE" | "NONE";
  default: string;
  arms: Record<string, ArmBundle>;
}

export interface ArmInfo {
  name: ArmName;
  trained_license: TrainedLicense;
  trained_at: string;
}

const MANIFEST = shipped as unknown as Manifest;
const TABLE = zipfTable as unknown as ZipfTable;

function float32s(base64: string): Float32Array {
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  return Float32Array.from({ length: bytes.length / 4 }, (_, i) => view.getFloat32(i * 4, true));
}

interface DenseLayer {
  rows: number;
  cols: number;
  weight: Float32Array;
  bias: Float32Array;
  activation: Layer["activation"];
}

const ACTIVATIONS = {
  relu: (x: number) => Math.max(x, 0),
  none: (x: number) => x,
  sigmoid: (x: number) => 1 / (1 + Math.exp(-x)),
};

const asArray = (queries: string | string[]) => (typeof queries === "string" ? [queries] : queries);
const clip01 = (x: number) => Math.min(Math.max(x, 0), 1);

class ArmModel {
  readonly arm: string;
  readonly trainedLicense: TrainedLicense;
  readonly trainedAt: string;
  readonly delta: number;
  readonly k: number;
  private readonly layers: DenseLayer[];
  private readonly sparse: number;
  private readonly dense: number;

  constructor(
    readonly bundle: ArmBundle,
    private readonly table: ZipfTable,
  ) {
    this.arm = bundle.arm;
    this.trainedLicense = bundle.trained_license;
    this.trainedAt = bundle.trained_at;
    this.delta = bundle.rrf_delta;
    this.k = bundle.fusion_k;
    this.sparse = bundle.head_routes.indexOf("sparse_only");
    this.dense = bundle.head_routes.indexOf("dense_only");
    this.layers = bundle.layers.map((l) => ({
      rows: l.shape[0],
      cols: l.shape[1],
      weight: float32s(l.weight),
      bias: float32s(l.bias),
      activation: l.activation,
    }));
  }

  /** Head probabilities in `head_routes` order. */
  private probabilities(text: string): number[] {
    const { zipf_mean: mean, zipf_std: std, zipf_rare: rare } = this.bundle;
    let x = zipfFeatures(text, this.table, rare).map((f, i) => (f - mean[i]) / std[i]);
    for (const { rows, cols, weight, bias, activation } of this.layers) {
      const act = ACTIVATIONS[activation];
      x = Array.from({ length: rows }, (_, r) => {
        let sum = bias[r];
        for (let c = 0; c < cols; c++) sum += weight[r * cols + c] * x[c];
        return act(sum);
      });
    }
    return x;
  }

  /** Most probable head among those clearing their threshold, else the most probable head. */
  private serve(probs: number[]): Route {
    const { thresholds, head_routes: heads, hedge, hedge_bar: hedgeBar } = this.bundle;
    const argmax = (idx: number[]) => idx.reduce((best, i) => (probs[i] > probs[best] ? i : best));
    const all = probs.map((_, i) => i);
    const fired = all.filter((i) => probs[i] >= thresholds[i]);
    if (fired.length) return heads[argmax(fired)];
    return Math.max(...probs) < hedgeBar ? hedge : heads[argmax(all)];
  }

  classify(queries: string | string[], { delta = this.delta }: { delta?: number } = {}): Classification[] {
    return asArray(queries).map((query) => {
      const p = this.probabilities(query);
      const [pSparse, pDense] = [p[this.sparse], p[this.dense]];
      const route = Math.abs(pDense - pSparse) < delta ? this.bundle.hedge : this.serve(p);
      return { query, route, p_dense: pDense, p_sparse: pSparse };
    });
  }

  classifyPure(queries: string | string[], options: { delta?: number } = {}): Route[] {
    return this.classify(queries, options).map((c) => c.route);
  }

  /** Each head's probability re-centred on its own threshold and scaled by the confidence dial k. */
  weighted(
    queries: string | string[],
    { k = this.k, delta = this.delta }: { k?: number; delta?: number } = {},
  ): Weighted[] {
    const [tS, tD] = [this.bundle.thresholds[this.sparse], this.bundle.thresholds[this.dense]];
    return asArray(queries).map((query) => {
      const p = this.probabilities(query);
      const [s, d] = [p[this.sparse], p[this.dense]];
      if (Math.abs(d - s) < delta) return { query, alpha: 0.5, p_dense: d, p_sparse: s };
      const [firesS, firesD] = [s >= tS, d >= tD];
      const sparseWins = firesS !== firesD ? firesS : s >= d;
      const [prob, bar] = sparseWins ? [s, tS] : [d, tD];
      const lean = clip01(0.5 + (0.5 * (prob - bar)) / (1 - bar));
      const anchored = sparseWins ? lean : 1 - lean;
      return { query, alpha: clip01(0.5 + k * (anchored - 0.5)), p_dense: d, p_sparse: s };
    });
  }
}

/** Routes queries with one of the shipped arms; without `arm`, the maintainer's default. */
export class Router {
  private arm: ArmModel;
  private readonly loaded = new Map<string, ArmModel>();

  constructor({ arm = MANIFEST.default as ArmName }: { arm?: ArmName } = {}) {
    this.arm = this.load(arm);
  }

  private load(name: string): ArmModel {
    const bundle = MANIFEST.arms[name];
    if (!bundle) throw new Error(`no arm ${name}; this build ships ${Object.keys(MANIFEST.arms).join(", ")}`);
    let arm = this.loaded.get(name);
    if (!arm) this.loaded.set(name, (arm = new ArmModel(bundle, TABLE)));
    return arm;
  }

  getArms(): ArmInfo[] {
    return Object.values(MANIFEST.arms).map((a) => ({
      name: a.arm as ArmName,
      trained_license: a.trained_license,
      trained_at: a.trained_at,
    }));
  }

  get current(): ArmName {
    return this.arm.arm as ArmName;
  }

  /** The licence level this build was exported at. */
  get allowLicense(): Manifest["allow_license"] {
    return MANIFEST.allow_license;
  }

  choose(name: ArmName): this {
    this.arm = this.load(name);
    return this;
  }

  classify(queries: string | string[], options: { delta?: number } = {}): Classification[] {
    return this.arm.classify(queries, options);
  }

  classifyPure(queries: string | string[], options: { delta?: number } = {}): Route[] {
    return this.arm.classifyPure(queries, options);
  }

  weighted(queries: string | string[], options: { k?: number; delta?: number } = {}): Weighted[] {
    return this.arm.weighted(queries, options);
  }
}
