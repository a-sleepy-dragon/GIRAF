import os
import sys
import time
from dataclasses import dataclass

from neo4j import GraphDatabase

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_MUTI = os.path.join(_ROOT, "multi-agent")
_TOOL = os.path.join(_ROOT, "tool")
_TAG = os.environ.get("WORKLOAD_ID", "x")
for _p in (_MUTI, _TOOL):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from estimate import estimate_index_storage


@dataclass(frozen=True)
class GraphIndex:
    on: str
    label: str
    props: tuple


def _q(name):
    return "`" + str(name).replace("`", "") + "`"


def _ddl(index, name):
    var = "r" if index.on == "relationship" else "n"
    props = ", ".join(f"{var}.{_q(p)}" for p in index.props)
    if index.on == "relationship":
        return f"CREATE RANGE INDEX {_q(name)} IF NOT EXISTS FOR ()-[r:{_q(index.label)}]-() ON ({props})"
    return f"CREATE RANGE INDEX {_q(name)} IF NOT EXISTS FOR (n:{_q(index.label)}) ON ({props})"


def anchors_from_explain(driver, queries):
    import baseline_anchor
    anchors = []
    with driver.session() as s:
        for query in queries:
            plan = s.run("EXPLAIN " + query["text"]).consume().plan
            baseline_anchor.collect_anchors(
                baseline_anchor._plan(plan), query["id"], anchors)
    candidates = {}
    for row in anchors:
        props = tuple(row.get("props") or [])
        if row.get("label") and props:
            idx = GraphIndex(row.get("on", "node"), row["label"], props)
            candidates[(idx.on, idx.label, idx.props)] = idx
    return sorted(candidates.values(), key=lambda x: (x.on, x.label, x.props))


class GraphSelectionAlgorithm:
    def __init__(self, database_connector, parameters=None):
        self.parameters = parameters or {}
        self.driver = database_connector
        if isinstance(database_connector, str):
            self.driver = GraphDatabase.driver(
                database_connector,
                auth=(self.parameters.get("user", "neo4j"),
                      self.parameters.get("password", "your_password")))
        self.budget_mb = float(self.parameters.get("budget_MB", 2048))
        self.max_indexes = int(self.parameters.get("max_indexes", 15))
        self.runs = max(1, int(self.parameters.get("runs", 3)))

    def _start(self, queries, tag):
        candidates = anchors_from_explain(self.driver, queries)
        print(f"[{tag}] start: {len(candidates)} candidates, "
              f"budget {self.budget_mb:.1f}MB, max_indexes={self.max_indexes}", flush=True)
        return candidates

    def _time(self, queries):
        result = {}
        with self.driver.session() as s:
            for q in queries:
                values = []
                for _ in range(self.runs):
                    t = time.perf_counter()
                    s.run(q["text"]).consume()
                    values.append((time.perf_counter() - t) * 1000)
                values.sort()
                if len(values) >= 3:
                    values = values[1:-1]
                result[q["id"]] = sum(values) / len(values)
        return result

    def _online(self, s, name):
        for _ in range(1200):
            row = s.run("SHOW INDEXES YIELD name,state WHERE name=$n RETURN state", n=name).single()
            if row and row["state"] == "ONLINE":
                return
            time.sleep(.25)
        raise TimeoutError(name)

    def _trial(self, candidate, active, queries, baseline, serial):
        print(f"    [trial] try {candidate.label}({','.join(candidate.props)}) "
              f"(selected {len(active)})", flush=True)
        names = []
        with self.driver.session() as s:
            for i, idx in enumerate(active):
                n = f"codex_w{_TAG}_h_{serial}_a{i}"
                s.run(_ddl(idx, n)).consume(); self._online(s, n); names.append(n)
            name = f"codex_w{_TAG}_h_{serial}_c"
            s.run(_ddl(candidate, name)).consume(); self._online(s, name); names.append(name)
            try:
                after = self._time(queries)
            finally:
                for n in reversed(names):
                    s.run(f"DROP INDEX {_q(n)} IF EXISTS").consume()
        gain = sum(max(0, baseline[q["id"]] - after[q["id"]]) for q in queries)
        size = float(estimate_index_storage(candidate.label, list(candidate.props), self.driver, candidate.on) or 0)
        return gain, size, after

    def _measure_configuration(self, indexes, queries, serial):
        print(f"    [cfg] measure {len(indexes)} indexes", flush=True)
        names = []
        with self.driver.session() as s:
            for i, idx in enumerate(indexes):
                name = f"codex_w{_TAG}_cfg_{serial}_{i}"
                s.run(_ddl(idx, name)).consume(); self._online(s, name); names.append(name)
            try:
                return self._time(queries)
            finally:
                for name in reversed(names):
                    s.run(f"DROP INDEX {_q(name)} IF EXISTS").consume()

    def _trial_drop(self, candidate, chosen, queries):
        print(f"    [drop-trial] try dropping {candidate.label}({','.join(candidate.props)})", flush=True)
        remaining = [x for x in chosen if x != candidate]
        names = []
        with self.driver.session() as s:
            for i, idx in enumerate(remaining):
                name = f"codex_w{_TAG}_drop_{i}"
                s.run(_ddl(idx, name)).consume()
                self._online(s, name)
                names.append(name)
            try:
                after = self._time(queries)
            finally:
                for name in reversed(names):
                    s.run(f"DROP INDEX {_q(name)} IF EXISTS").consume()
        return sum(after[q["id"]] for q in queries)

    def _greedy(self, candidates, queries, tag):
        selected, baseline, used = [], self._time(queries), 0.0
        deadline = time.time() + float(self.parameters.get("max_runtime_minutes", 1)) * 60
        print(f"  [{tag}] baseline measured, start greedy rounds ({len(candidates)} candidates)", flush=True)
        while candidates and len(selected) < self.max_indexes:
            if time.time() >= deadline:
                break
            print(f"  [{tag}] round {len(selected)+1}: {len(candidates)} candidates left", flush=True)
            trials = []
            for n, c in enumerate(candidates):
                gain, size, after = self._trial(c, selected, queries, baseline, n)
                if size > 0 and used + size <= self.budget_mb:
                    trials.append((gain / size, c, size, gain, after))
            if not trials:
                break
            best = max(trials, key=lambda x: x[0])
            if best[3] <= float(self.parameters.get("min_gain_ms", 0)):
                break
            _, c, size, _, baseline = best
            selected.append(c); used += size; candidates.remove(c)
        return selected

    def calculate_best_indexes(self, queries):
        raise NotImplementedError
