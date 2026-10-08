// The JSON API served by `latentpair serve` (see ../README.md).

export interface Info {
  model_path: string;
  nodes: number;
  cells: number;
  depth: number;
  radices: number[];
  read: { texts: number; units: number; contexts_used: number };
  judged: { texts: number; rewards_total: number; penalties_total: number };
  settings: { outcomes: number; alpha: number; floor: number; smoothing: number; rungs: string; backoff: string };
  tokenizer: { levels: string; code_bits: number; window: number; steps: number; params: number };
  training: string | null;
}

export interface PathResult {
  text: string;
  full_text: string;
  units: number[];
  unit_names: string[];
  cost: number;
  step_costs: number[];
  traversal: string;
  mode: string;
  reached_end: boolean;
  peak: number;
  bits_per_unit: number;
}

export interface FoldResult {
  code: number[];
  levels: { level: number; node: number; seen: number; own: number; decode: string }[];
  top: { unit: string; x: number; p: number }[];
  entropy_bits: number;
}

export interface JudgeRecord {
  call: string;
  cells?: number;
  rewarded?: number;
  punished?: number;
  strength?: number;
  outcomes?: number;
  amount?: number;
  prefix?: string;
}

export interface JudgeResult {
  record: JudgeRecord;
  changes: { unit: string; before: number; after: number }[];
}

export interface Score {
  bits: number;
  mean_reward: number;
  worst_penalty: number;
  units: number;
  per_unit: { unit: string; bits: number; reward: number; penalty: number }[];
}

export interface Stat {
  step: number;
  loss_bits: number;
  next_bits: number[];
  accuracy: number[];
  tail: number[];
  train_bits: number;
  lr: number;
  seconds: number;
}

export interface Progress {
  running: boolean;
  done: boolean;
  error: string | null;
  stats: Stat[];
  params: number;
}

export interface Classes {
  radix: number;
  total: number;
  classes: { symbol: number; count: number; share: number; prototype: string; example: string }[];
}

export async function api<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const text = await res.text();
  let data: { error?: string } & T;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(text || `${res.status} ${res.statusText}`);
  }
  if (!res.ok || data.error) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}
