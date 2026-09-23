// Port of the router's five wordfreq rarity scalars (encoder_router.table.ZipfStats) over the
// exported English table; English only, so it matches the Python path for English queries.

export interface ZipfTable {
  centizipf: Record<string, number>;
  smashed_freq: Record<string, number>;
}

const WORD = /[a-z0-9]+/g;
const MULTI_DIGIT = /\d[\d.,]+/g;
const DIGIT_FREQS = [0.009, 0.3, 0.175, 0.124, 0.096, 0.078, 0.066, 0.057, 0.05, 0.045];
const YEAR_LOG_PEAK = -1.9185;
const NOT_YEAR_PROB = 0.1;
const REFERENCE_YEAR = 2019;
const PLATEAU_WIDTH = 20;
const MIN_FREQ = 1e-9; // zipf 0, wordfreq's floor

// Python's math.log(x, 10) is ln(x)/ln(10), not log10 — they can differ in the last bit.
const log10 = (x: number) => Math.log(x) / Math.log(10);
// Python's round(x, n) rounds the exact binary value; toFixed does the same (exact ties aside).
const round = (x: number, digits: number) => Number(x.toFixed(digits));

function benfordFreq(digits: string): number {
  return DIGIT_FREQS[Number(digits[0])] / 10 ** (digits.length - 1);
}

function yearFreq(digits: string): number {
  const year = Number(digits);
  let logFreq: number;
  if (year <= REFERENCE_YEAR) logFreq = YEAR_LOG_PEAK - 0.0083 * (REFERENCE_YEAR - year);
  else if (year <= REFERENCE_YEAR + PLATEAU_WIDTH) logFreq = YEAR_LOG_PEAK;
  else logFreq = YEAR_LOG_PEAK - 0.2 * (year - (REFERENCE_YEAR + PLATEAU_WIDTH));
  return 10.0 ** logFreq + NOT_YEAR_PROB * benfordFreq(digits);
}

function digitFreq(token: string): number {
  let freq = 1.0;
  for (const run of token.match(MULTI_DIGIT) ?? []) {
    for (const digits of run.match(/\d+/g) ?? []) {
      freq *= digits.length === 4 ? yearFreq(digits) : benfordFreq(digits);
    }
  }
  return freq;
}

/** wordfreq's `zipf_frequency(token, "en")` for one `[a-z0-9]+` token. */
export function zipf(token: string, table: ZipfTable): number {
  const smashed = token.replace(MULTI_DIGIT, (run) => run.replace(/\d/g, "0"));
  if (smashed === token) return (table.centizipf[token] ?? 0) / 100;
  const base = table.smashed_freq[smashed];
  if (base === undefined) return 0;
  const freq = Math.max(base * digitFreq(token), MIN_FREQ);
  const rounded = round(freq, Math.floor(-log10(freq)) + 3); // wordfreq keeps 3 significant digits
  return round(log10(rounded) + 9, 2) + 0; // + 0 turns -0 into 0
}

/** [min, mean, max, rare share, OOV share] of the query's token rarities; zeros when it has no tokens. */
export function zipfFeatures(text: string, table: ZipfTable, rare: number): number[] {
  const tokens = text.toLowerCase().match(WORD);
  if (!tokens) return [0, 0, 0, 0, 0];
  const freqs = tokens.map((t) => zipf(t, table));
  const n = freqs.length;
  const sum = freqs.reduce((a, b) => a + b, 0);
  return [
    Math.min(...freqs),
    sum / n,
    Math.max(...freqs),
    freqs.filter((f) => f < rare).length / n,
    freqs.filter((f) => f === 0).length / n,
  ].map(Math.fround); // the Python path stores these as float32
}
