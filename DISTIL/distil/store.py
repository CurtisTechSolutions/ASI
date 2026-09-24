"""Where the vectors actually live.

The in-process store is a linear scan. That is a deliberate default, not an
oversight: 512 dims x 100k traces is ~50M multiply-adds, which pure Python does
in a few seconds and which is *exact*. Every approximate index trades recall for
speed, and the trade only pays once the scan is the bottleneck. It is not the
bottleneck while a model round trip sits in front of it.

Past that, the backend swaps. `TraceStore` is the seam, and there are two real
implementations of it behind lazy imports so this package keeps its zero
dependencies:

**Postgres + pgvector.** The best fit, because the thing that makes this system
work is not the nearest-neighbour search -- it is that recall ranks by
`similarity x credibility x recency`, and in Postgres all three are columns of
the same row. The whole composite score is one SQL expression, grades update
transactionally in the same place the vectors live, and `kind` filters are a
WHERE clause the planner understands. The DDL and the scoring query are written
out in `PostgresStore` verbatim.

**Redis.** RediSearch KNN returns strictly by vector distance, so the composite
score has to be applied client-side after an over-fetch. That is fine, and it is
the right choice when the store is shared between processes and write latency
matters more than ranking exactness.

One caveat that applies to every approximate index and is easy to get wrong:

> ANN returns the top-k by *vector distance*. This system then re-ranks by
> credibility. A verified, slightly-less-similar trace that fell outside the ANN
> top-k can never be re-ranked back into the answer -- it was never fetched.

So approximate stores declare `approximate = True` and `Memory` over-fetches by
`OVERFETCH` before re-ranking. The exact store does not need it. Getting this
wrong produces a system that silently stops surfacing its best-verified memories
as the store grows, which is the failure mode that would take longest to notice.
"""
from __future__ import annotations

import json
import os
from typing import Iterable

from .vector import Vector, cosine

OVERFETCH = 10          # multiplier applied to k on approximate backends


class TraceStore:
    """The seam. Implementations hold `Trace` objects and answer vector queries.

    `search` returns (trace, similarity) and does no credibility or recency
    weighting -- that belongs to `Memory`, so every backend ranks identically and
    a backend swap cannot silently change what the system recalls.
    """

    approximate = False

    def put(self, trace) -> None: raise NotImplementedError
    def get(self, trace_id: str): raise NotImplementedError
    def delete(self, trace_id: str) -> bool: raise NotImplementedError
    def all(self) -> Iterable: raise NotImplementedError
    def count(self) -> int: raise NotImplementedError
    def touch(self, trace) -> None: """Persist mutated counters/grades."""
    def search(self, vector: Vector, k: int, kinds=None, min_similarity: float = 0.0,
               exclude: set[str] | None = None, embedder: str | None = None) -> list[tuple]:
        raise NotImplementedError


class InProcessStore(TraceStore):
    """Exact search over a dict. The default, and what the tests run against."""

    approximate = False

    def __init__(self) -> None:
        self._traces: dict[str, object] = {}
        self._order: list[str] = []

    def put(self, trace) -> None:
        if trace.id not in self._traces:
            self._order.append(trace.id)
        self._traces[trace.id] = trace

    def get(self, trace_id: str):
        return self._traces.get(trace_id)

    def delete(self, trace_id: str) -> bool:
        if trace_id not in self._traces:
            return False
        del self._traces[trace_id]
        self._order.remove(trace_id)
        return True

    def all(self):
        return [self._traces[i] for i in self._order if i in self._traces]

    def count(self) -> int:
        return len(self._traces)

    def touch(self, trace) -> None:
        self._traces[trace.id] = trace          # same object; nothing to write back

    def search(self, vector, k, kinds=None, min_similarity=0.0, exclude=None, embedder=None):
        exclude = exclude or set()
        out = []
        for t in self._traces.values():
            if t.id in exclude or (kinds and t.kind not in kinds):
                continue
            if embedder and t.embedder != embedder:
                continue
            sim = cosine(t.vector, vector)
            if sim >= min_similarity:
                out.append((t, sim))
        out.sort(key=lambda pair: pair[1], reverse=True)
        return out[:k] if k else out


class PostgresStore(TraceStore):
    """pgvector. Requires `psycopg` (v3) and the `vector` extension.

    Not exercised by this package's test suite -- there is no database in the
    environment it was built in -- so it is written to be read as much as run.
    Schema:

        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE distil_traces (
            id          text PRIMARY KEY,
            kind        text NOT NULL,
            text        text NOT NULL,
            embedder    text NOT NULL,
            created     double precision NOT NULL,
            last_used   double precision NOT NULL DEFAULT 0,
            hits        integer NOT NULL DEFAULT 0,
            seen        integer NOT NULL DEFAULT 1,
            meta        jsonb NOT NULL DEFAULT '{}',
            links       jsonb NOT NULL DEFAULT '[]',
            grades      jsonb NOT NULL DEFAULT '[]',
            grade_n     integer NOT NULL DEFAULT 0,   -- denormalised for scoring
            grade_sum   double precision NOT NULL DEFAULT 0,
            embedding   vector(512) NOT NULL
        );
        CREATE INDEX ON distil_traces USING hnsw (embedding vector_cosine_ops);
        CREATE INDEX ON distil_traces (kind);

    `grade_n` / `grade_sum` are denormalised out of the `grades` array on write
    so the composite score is computable without unnesting JSON per row. The
    whole ranking is then one statement -- similarity, shrunk credibility and
    exponential recency, exactly as `Memory.recall` computes them:

        SELECT id, 1 - (embedding <=> %(q)s) AS sim,
               (%(prior)s * %(m)s + grade_sum) / (%(m)s + grade_n) AS cred,
               POWER(0.5, GREATEST(0, %(now)s - GREATEST(created, last_used))
                          / 86400.0 / %(halflife)s) AS rec
        FROM distil_traces
        WHERE embedder = %(embedder)s AND (%(kinds)s IS NULL OR kind = ANY(%(kinds)s))
        ORDER BY POWER(1 - (embedding <=> %(q)s), %(w_sim)s)
               * POWER((%(prior)s * %(m)s + grade_sum) / (%(m)s + grade_n), %(w_cred)s)
               * POWER(POWER(0.5, GREATEST(0, %(now)s - GREATEST(created, last_used))
                        / 86400.0 / %(halflife)s), %(w_rec)s) DESC
        LIMIT %(k)s;

    Pushing the score into SQL is the reason to prefer Postgres here, and the
    reason this class still returns bare similarities: a mixed deployment where
    one backend ranks server-side and another ranks client-side would rank
    differently, and a memory layer whose answers depend on where it is hosted
    is not a memory layer. Ranking stays in `Memory`; the SQL above is what to
    use when the scan becomes the bottleneck and that consistency is worth
    re-verifying against.
    """

    approximate = True          # HNSW is approximate; Memory over-fetches

    def __init__(self, dsn: str, table: str = "distil_traces") -> None:
        import psycopg                       # lazy: keeps the package dependency-free
        self.psycopg = psycopg
        self.dsn = dsn
        self.table = table
        self.conn = psycopg.connect(dsn, autocommit=True)

    @staticmethod
    def _lit(v: Vector) -> str:
        return "[" + ",".join(f"{x:.6g}" for x in v) + "]"

    def put(self, trace) -> None:
        g = trace.grades
        with self.conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO {self.table}
                    (id, kind, text, embedder, created, last_used, hits, seen,
                     meta, links, grades, grade_n, grade_sum, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                      last_used=EXCLUDED.last_used, hits=EXCLUDED.hits, seen=EXCLUDED.seen,
                      meta=EXCLUDED.meta, links=EXCLUDED.links, grades=EXCLUDED.grades,
                      grade_n=EXCLUDED.grade_n, grade_sum=EXCLUDED.grade_sum,
                      -- text and embedding too: a trace can be re-embedded in
                      -- place (Toolbox.record_use does exactly that), and an
                      -- upsert that skips them would persist the counters while
                      -- silently keeping the stale description and vector.
                      text=EXCLUDED.text, embedding=EXCLUDED.embedding""",
                (trace.id, trace.kind, trace.text, trace.embedder, trace.created,
                 trace.last_used, trace.hits, trace.seen, json.dumps(trace.meta),
                 json.dumps(trace.links), json.dumps([list(x) for x in g]),
                 len(g), sum((s + 1.0) / 2.0 for s, _, _ in g), self._lit(trace.vector)))

    touch = put

    def get(self, trace_id: str):
        from .memory import Trace
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id,kind,text,embedder,created,last_used,hits,seen,"
                        f"meta,links,grades,embedding FROM {self.table} WHERE id=%s", (trace_id,))
            row = cur.fetchone()
        return self._row(row) if row else None

    def delete(self, trace_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE id=%s", (trace_id,))
            return cur.rowcount > 0

    @staticmethod
    def _row(row):
        from .memory import Trace
        return Trace(id=row[0], kind=row[1], text=row[2], embedder=row[3], created=row[4],
                     last_used=row[5], hits=row[6], seen=row[7], meta=row[8], links=row[9],
                     grades=[tuple(x) for x in row[10]],
                     vector=[float(x) for x in json.loads(str(row[11]))])

    def all(self):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id,kind,text,embedder,created,last_used,hits,seen,"
                        f"meta,links,grades,embedding FROM {self.table}")
            return [self._row(r) for r in cur.fetchall()]

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return cur.fetchone()[0]

    def search(self, vector, k, kinds=None, min_similarity=0.0, exclude=None, embedder=None):
        clauses, params = [], {"q": self._lit(vector), "k": max(k, 1)}
        if embedder:
            clauses.append("embedder = %(embedder)s")
            params["embedder"] = embedder
        if kinds:
            clauses.append("kind = ANY(%(kinds)s)")
            params["kinds"] = list(kinds)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.conn.cursor() as cur:
            cur.execute(f"""SELECT id,kind,text,embedder,created,last_used,hits,seen,
                                   meta,links,grades,embedding, 1-(embedding <=> %(q)s) AS sim
                            FROM {self.table} {where}
                            ORDER BY embedding <=> %(q)s LIMIT %(k)s""", params)
            rows = cur.fetchall()
        exclude = exclude or set()
        return [(self._row(r[:12]), float(r[12])) for r in rows
                if r[0] not in exclude and float(r[12]) >= min_similarity]


class RedisStore(TraceStore):
    """RediSearch vector index. Requires `redis`.

    Also not exercised here. Index:

        FT.CREATE distil ON HASH PREFIX 1 distil: SCHEMA
            kind TAG  embedder TAG  text TEXT
            embedding VECTOR HNSW 6 TYPE FLOAT32 DIM 512 DISTANCE_METRIC COSINE

    Redis cannot express the composite score in the query, so ranking is
    client-side after an over-fetch -- which is what `approximate = True` buys.
    Choose this when several processes share one memory and write latency
    dominates; choose Postgres when ranking quality and transactional grade
    updates dominate.
    """

    approximate = True

    def __init__(self, url: str = "redis://localhost:6379", index: str = "distil",
                 prefix: str = "distil:", ttl_seconds: int | None = None) -> None:
        import redis                         # lazy, as above
        self.redis = redis.Redis.from_url(url, decode_responses=False)
        self.index = index
        self.prefix = prefix
        self.ttl = ttl_seconds              # None = no expiry; set it for short-term memory

    @staticmethod
    def _pack(v: Vector) -> bytes:
        import struct
        return struct.pack(f"<{len(v)}f", *v)

    @staticmethod
    def _unpack(b: bytes) -> Vector:
        import struct
        return list(struct.unpack(f"<{len(b)//4}f", b))

    def put(self, trace) -> None:
        key = f"{self.prefix}{trace.id}"
        self.redis.hset(key, mapping={
            b"id": trace.id.encode(), b"kind": trace.kind.encode(),
            b"text": trace.text.encode(), b"embedder": trace.embedder.encode(),
            b"blob": json.dumps(trace.to_json()).encode(),
            b"embedding": self._pack(trace.vector)})
        if self.ttl:
            # Every write refreshes the clock, so the TTL measures time since a
            # trace was last *touched* rather than since it was created. That is
            # what makes expiry an access policy instead of a shredder: recall
            # calls `touch`, so anything still being used keeps resetting its own
            # week and only genuinely unused traces age out.
            self.redis.expire(key, self.ttl)

    touch = put

    def get(self, trace_id: str):
        from .memory import Trace
        blob = self.redis.hget(f"{self.prefix}{trace_id}", b"blob")
        return Trace.from_json(json.loads(blob)) if blob else None

    def delete(self, trace_id: str) -> bool:
        return bool(self.redis.delete(f"{self.prefix}{trace_id}"))

    def all(self):
        from .memory import Trace
        out = []
        for key in self.redis.scan_iter(match=f"{self.prefix}*"):
            blob = self.redis.hget(key, b"blob")
            if blob:
                out.append(Trace.from_json(json.loads(blob)))
        return out

    def count(self) -> int:
        return sum(1 for _ in self.redis.scan_iter(match=f"{self.prefix}*"))

    def search(self, vector, k, kinds=None, min_similarity=0.0, exclude=None, embedder=None):
        from .memory import Trace
        filters = []
        if kinds:
            filters.append("@kind:{" + "|".join(kinds) + "}")
        if embedder:
            filters.append("@embedder:{" + embedder.replace("-", "\\-") + "}")
        prefilter = " ".join(filters) if filters else "*"
        query = f"({prefilter})=>[KNN {max(k,1)} @embedding $vec AS dist]"
        raw = self.redis.execute_command(
            "FT.SEARCH", self.index, query, "PARAMS", "2", "vec", self._pack(vector),
            "SORTBY", "dist", "DIALECT", "2", "RETURN", "2", "blob", "dist")
        exclude, out = exclude or set(), []
        for i in range(1, len(raw), 2):                 # [count, key, [f, v, ...], ...]
            fields = {raw[i + 1][j]: raw[i + 1][j + 1] for j in range(0, len(raw[i + 1]), 2)}
            trace = Trace.from_json(json.loads(fields[b"blob"]))
            sim = 1.0 - float(fields[b"dist"])          # COSINE distance -> similarity
            if trace.id not in exclude and sim >= min_similarity:
                out.append((trace, sim))
        return out


class TieredStore(TraceStore):
    """Short-term memory with a TTL in front of long-term memory that keeps.

    Redis holds everything for a week; Postgres holds what earned its place. The
    split exists because this system writes far more than it should keep --
    every query, every reasoning chain, every discarded idea becomes a trace --
    and a store that retains all of it by default grows without bound while
    getting worse at recall. Tiering inverts the default: **expiry is automatic
    and retention is earned**.

    A trace is promoted to long-term when any of these holds:

    - its kind is one the system cannot afford to lose (tools, cases, games);
    - it has been graded, positively or negatively, because a refuted approach is
      as much a result as a working one;
    - it has been recalled again since it was written (`hits > 0`), which is the
      store's own evidence that something wanted it.

    Everything else expires quietly, which is the intended outcome for the
    thousands of traces that were never consulted twice.

    The alternative -- Redis purely as a read-through cache over Postgres -- is
    supported by `write_through=True`, and it is the wrong default here. Caching
    makes reads faster without making the store smaller; the problem this design
    actually has is unbounded low-value accumulation, and only the tiered version
    addresses it. Use `write_through` when several processes share one memory and
    every trace genuinely matters.

    Neither backend is exercised by the test suite -- there is no Redis and no
    Postgres in the environment this was built in -- so `InProcessStore` stands in
    for both in `test_tiered_promotion_rule`, which tests the promotion *rule*,
    the part that is easy to get wrong.
    """

    approximate = True

    WEEK = 7 * 24 * 3600

    def __init__(self, hot: TraceStore, cold: TraceStore, ttl_seconds: int = WEEK,
                 preserve: tuple[str, ...] = ("tool", "solution", "game"),
                 write_through: bool = False) -> None:
        self.hot = hot
        self.cold = cold
        self.ttl = ttl_seconds
        self.preserve = preserve
        self.write_through = write_through

    # -- the rule ------------------------------------------------------------

    def worth_keeping(self, trace) -> bool:
        if trace.kind in self.preserve:
            return True
        if trace.grades:
            return True
        return trace.hits > 0

    # -- reads and writes ----------------------------------------------------

    def put(self, trace) -> None:
        self.hot.put(trace)
        if self.write_through or self.worth_keeping(trace):
            self.cold.put(trace)

    def touch(self, trace) -> None:
        """Re-writes to both tiers when the trace has now earned long-term
        storage. Recall calls this, so a trace being read for the second time is
        exactly when it gets promoted -- which is the point."""
        self.hot.touch(trace)
        if self.write_through or self.worth_keeping(trace):
            self.cold.put(trace)

    def get(self, trace_id: str):
        return self.hot.get(trace_id) or self.cold.get(trace_id)

    def delete(self, trace_id: str) -> bool:
        return any([self.hot.delete(trace_id), self.cold.delete(trace_id)])

    def all(self):
        seen, out = set(), []
        for t in list(self.hot.all()) + list(self.cold.all()):
            if t.id not in seen:
                seen.add(t.id)
                out.append(t)
        return out

    def count(self) -> int:
        return len(self.all())

    def search(self, vector, k, kinds=None, min_similarity=0.0, exclude=None, embedder=None):
        """Both tiers, merged, deduplicated by id, best similarity kept.

        Searching both rather than falling through matters: the hot tier holds
        this week's work and the cold tier holds everything that lasted, and a
        query about a recurring problem wants both -- what just happened and what
        was learned about it last time.
        """
        merged: dict[str, tuple] = {}
        for store in (self.hot, self.cold):
            for trace, sim in store.search(vector, k, kinds, min_similarity, exclude, embedder):
                if trace.id not in merged or sim > merged[trace.id][1]:
                    merged[trace.id] = (trace, sim)
        out = sorted(merged.values(), key=lambda pair: pair[1], reverse=True)
        return out[:k] if k else out

    def consolidate(self) -> dict:
        """Sweep the hot tier and promote everything that has earned it.

        Runs as maintenance. Redis expiry does the forgetting on its own; this
        makes sure nothing that qualified was left behind because it earned its
        place between writes.
        """
        promoted = 0
        for trace in self.hot.all():
            if self.worth_keeping(trace) and self.cold.get(trace.id) is None:
                self.cold.put(trace)
                promoted += 1
        return {"promoted": promoted, "hot": self.hot.count(), "cold": self.cold.count()}


def tiered(redis_url: str = "redis://localhost:6379", dsn: str = "",
           ttl_seconds: int = TieredStore.WEEK, write_through: bool = False) -> TieredStore:
    """Convenience wiring: Redis short-term over Postgres long-term."""
    return TieredStore(hot=RedisStore(redis_url, ttl_seconds=ttl_seconds),
                       cold=PostgresStore(dsn), ttl_seconds=ttl_seconds,
                       write_through=write_through)


class StoreError(RuntimeError):
    """A backend was asked for and could not be built. The message is the fix."""


#: What `DISTIL_STORE` accepts, and what each needs.
BACKENDS = {
    "memory": "in-process exact search; no server, the default",
    "postgres": "pgvector; needs DATABASE_URL and the psycopg package",
    "redis": "Redis vector search; needs REDIS_URL and the redis package",
    "tiered": "Redis short-term over Postgres long-term; needs both",
}


def describe() -> dict:
    """Which backend is configured and what it would need. For `distil stores`."""
    kind = (os.environ.get("DISTIL_STORE") or "memory").strip().lower()
    return {"backend": kind, "known": kind in BACKENDS,
            "database_url": bool(os.environ.get("DATABASE_URL")),
            "redis_url": bool(os.environ.get("REDIS_URL")),
            "ttl_seconds": _ttl(),
            "options": BACKENDS}


def _ttl() -> int:
    raw = os.environ.get("DISTIL_REDIS_TTL", "")
    try:
        return int(raw) if raw else TieredStore.WEEK
    except ValueError:
        raise StoreError(f"DISTIL_REDIS_TTL must be a whole number of seconds, not {raw!r}")


def _need(var: str, what: str) -> str:
    value = os.environ.get(var, "").strip()
    if not value:
        raise StoreError(f"{what} needs {var}; set it, or unset DISTIL_STORE to use "
                         f"the in-process store")
    return value


def from_env(kind: str | None = None):
    """Build the store `DISTIL_STORE` asks for, or None for the default.

    Raises rather than falling back. A misconfigured backend that quietly
    degrades to in-process memory is the same failure as a provider that answers
    nothing while reporting healthy: the system keeps working, the data goes
    somewhere other than where you think, and nothing says so. If you asked for
    Postgres you want Postgres or an error.

    Neither Redis nor Postgres is exercised by the test suite -- there is no
    server in the environment this was built in -- so treat the first run against
    a real one as the actual test.
    """
    kind = (kind or os.environ.get("DISTIL_STORE") or "memory").strip().lower()
    if kind in ("", "memory", "inprocess", "in-process"):
        return None                       # Memory builds its own InProcessStore
    if kind not in BACKENDS:
        raise StoreError(f"unknown DISTIL_STORE {kind!r}; have {', '.join(sorted(BACKENDS))}")
    try:
        if kind == "postgres":
            return PostgresStore(_need("DATABASE_URL", "the postgres store"))
        if kind == "redis":
            return RedisStore(_need("REDIS_URL", "the redis store"), ttl_seconds=_ttl())
        return tiered(redis_url=_need("REDIS_URL", "the tiered store"),
                      dsn=_need("DATABASE_URL", "the tiered store"),
                      ttl_seconds=_ttl())
    except StoreError:
        raise
    except ImportError as exc:
        # The drivers are the one place this package is not standard-library
        # only, and they are optional on purpose. Say which one is missing.
        driver = "psycopg[binary]" if "psycopg" in str(exc) else "redis"
        raise StoreError(f"the {kind} store needs the {driver} package: "
                         f"pip install {driver}") from exc
    except Exception as exc:
        raise StoreError(f"could not connect the {kind} store: {exc}") from exc
