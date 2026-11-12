"""Benchmark the recommended indexes in report_workload_N.json: before vs after index creation.
Usage: python benchmark.py --workload 1
"""
import argparse
import json
import os
import time

import config
from neo4j import GraphDatabase

import loaders

MB = 1024 * 1024


def btick(name):
    return "`" + str(name).replace("`", "") + "`"


def load_report(workload_id):
    path = os.path.join(config.OUT_DIR, f"report_workload_{workload_id}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def show_user_indexes(session):
    """Return the list of current non-built-in user index names."""
    recs = session.run("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name").data()
    return [r["name"] for r in recs]


def create_index(session, name, label, props, on="node"):
    cols = ", ".join(f"{'r' if on == 'relationship' else 'n'}.{btick(p)}" for p in props)
    if on == "relationship":
        session.run(f"CREATE RANGE INDEX `{name}` IF NOT EXISTS "
                    f"FOR ()-[r:{btick(label)}]-() ON ({cols})")
    else:
        session.run(f"CREATE RANGE INDEX `{name}` IF NOT EXISTS "
                    f"FOR (n:{btick(label)}) ON ({cols})")


def wait_online(session, name, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        rec = session.run("SHOW INDEXES YIELD name, state WHERE name = $n", n=name).single()
        if rec and rec["state"] == "ONLINE":
            return
        time.sleep(0.5)
    raise TimeoutError(f"index {name} did not become ONLINE within {timeout}s")


def drop_index(session, name):
    session.run(f"DROP INDEX `{name}` IF EXISTS")


def run_timed(session, cypher):
    """Run a query and return server time (ms) / first-result time (ms) / wall-clock (ms)."""
    t0 = time.perf_counter()
    res = session.run(cypher)
    summary = res.consume()
    wall = (time.perf_counter() - t0) * 1000.0
    return {
        "server_ms": summary.result_consumed_after,
        "available_ms": summary.result_available_after,
        "wall_ms": wall,
    }


def trimmed_mean(times):
    times = sorted(times)
    if len(times) >= 3:
        times = times[1:-1]
    return sum(times) / len(times)


def measure(session, queries, runs=5):
    """Run each query `runs` times and return {qid: {times, trimmed_mean, ...}}.

    Removing max and min already drops the cold-start first run (usually the max), so no separate warm-up.
    """
    out = {}
    for q in queries:
        times = []
        for _ in range(runs):
            times.append(run_timed(session, q["text"])["wall_ms"])
        out[q["id"]] = {
            "times_ms": [round(t, 2) for t in times],
            "trimmed_mean_ms": round(trimmed_mean(times), 2),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5, help="how many times each query runs (mean after removing max/min)")
    args = ap.parse_args()

    report = load_report(args.workload)
    rec_indexes = report.get("recommended_indexes", [])
    if not rec_indexes:
        raise SystemExit("no recommended_indexes in the report, cannot test")
    print(f"Indexes to test (from report_workload_{args.workload}.json):")
    for idx in rec_indexes:
        kind = "relationship" if idx.get("on") == "relationship" else "node"
        print(f"  - [{kind}] {idx['label']}({','.join(idx['props'])}) type={idx.get('type','?')} "
              f"cost={idx.get('cost_mb','?')}MB")

    queries = loaders.load_workload(args.workload)
    print(f"workload queries: {', '.join(q['id'] for q in queries)}\n")

    driver = GraphDatabase.driver(config.NEO4J_URI, auth=config.NEO4J_AUTH)
    try:
        with driver.session() as s:
            before_idx = show_user_indexes(s)
            if before_idx:
                print(f"[NOTE] user indexes already exist before testing: {before_idx} (should only be built-in LOOKUP)")

            print("=== Phase A: no indexes (baseline) ===")
            base = measure(s, queries, runs=args.runs)
            for qid, m in base.items():
                print(f"  {qid}: 5 runs={m['times_ms']} trim mean={m['trimmed_mean_ms']}ms")

            print("\n=== Create indexes ===")
            idx_names = []
            for idx in rec_indexes:
                label, props = idx["label"], idx["props"]
                on = idx.get("on", "node")
                name = "bench_" + label.lower() + "_" + "_".join(props).lower()
                t0 = time.time()
                create_index(s, name, label, props, on)
                wait_online(s, name)
                idx_names.append(name)
                kind = "relationship" if on == "relationship" else "node"
                print(f"  + {name} ON {label}({','.join(props)}) [{kind}] "
                      f"({time.time()-t0:.1f}s ONLINE)")

            print("\n=== Phase B: after index creation ===")
            after = measure(s, queries, runs=args.runs)
            for qid, m in after.items():
                print(f"  {qid}: 5 runs={m['times_ms']} trim mean={m['trimmed_mean_ms']}ms")

            print("\n=== Drop indexes to restore ===")
            for name in idx_names:
                drop_index(s, name)
                print(f"  - DROP {name}")
            remaining = show_user_indexes(s)
            print(f"  remaining user indexes: {remaining or 'none'}")

        print("\n=== Before/after comparison (mean after removing max/min, unit ms) ===")
        print(f"{'query':8s} {'no-index':>10s} {'indexed':>10s} {'speedup':>8s}")
        summary_rows = []
        for q in queries:
            qid = q["id"]
            b, a = base[qid]["trimmed_mean_ms"], after[qid]["trimmed_mean_ms"]
            speedup = round(b / a, 3) if a > 0 else float("inf")
            print(f"{qid:8s} {b:>10.2f} {a:>10.2f} {speedup:>7.2f}x")
            summary_rows.append({"query": qid, "before_ms": b, "after_ms": a,
                                 "speedup": speedup})

        result = {
            "workload": args.workload,
            "indexes": rec_indexes,
            "per_query": summary_rows,
            "detail": {"before": base, "after": after},
        }
        out_path = os.path.join(config.OUT_DIR, f"bench_workload_{args.workload}.json")
        os.makedirs(config.OUT_DIR, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nresult written: {out_path}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
