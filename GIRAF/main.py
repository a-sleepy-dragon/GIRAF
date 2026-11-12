# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import time

import config
from neo4j import GraphDatabase

import loaders
import rag
import agents
import orchestrator
import benchmark
import llm_client

sys.path.insert(0, config.TOOL_DIR)
from estimate_index_storage import estimate_index_storage as _v2_est
import estimate as _estimate_mod


def _opname(op):
    return (op or "").split("@")[0]


def _plan(node):
    if isinstance(node, dict):
        return node
    if node is not None and hasattr(node, "to_dict"):
        return node.to_dict()
    return None


def render_plan(plan, max_detail=90):
    lines = []

    def walk(node, depth):
        node = _plan(node)
        if not isinstance(node, dict):
            return
        op = _opname(node.get("operatorType", ""))
        args = node.get("args") or node.get("arguments") or {}
        est = args.get("EstimatedRows", "?")
        det = args.get("Details", "") or ""
        if len(det) > max_detail:
            det = det[:max_detail] + "..."
        line = "  " * depth + f"- {op} [est={est}]"
        if det:
            line += f"  {det}"
        lines.append(line)
        for c in node.get("children", []) or []:
            walk(c, depth + 1)

    walk(plan, 0)
    return "\n".join(lines)


def _key(r):
    return agents.candidate_key(r["label"], r["props"], r.get("on", "node"))


def _make_est_v2(alpha):
    def _est_v2(label, props, driver=None, on="node"):
        kind = "relationship" if on == "relationship" else "node"
        return _v2_est(label, list(props), driver, kind=kind, internal_factor=alpha)
    return _est_v2


def apply_dataset(ds):
    config.DATASET_DIR = os.path.join(config.PROJECT_ROOT, "datasets", ds["dataset_dir"])
    if ds["single_prop_sets"]:
        config.SINGLE_PROP_SETS_JSON = os.path.join(config.HERE, ds["single_prop_sets"])
    config.RAG_KB_DIR = os.path.join(config.HERE, "rag_kb2")
    config.RAG_ENABLED = True
    config.RAG_TOP_K = 4
    config.TEMPERATURE = 0.0
    _estimate_mod.estimate_index_storage = _make_est_v2(ds["alpha"])


def resolve_workloads(ds):
    w = ds["workloads"]
    if w == "auto" or w is None:
        return [x for x in range(1, 31)
                if os.path.isdir(os.path.join(config.DATASET_DIR, f"workload_{x}"))]
    return list(w)


def load_schema_for(ds):
    if ds["schema_path"]:
        with open(os.path.join(config.HERE, ds["schema_path"]), encoding="utf-8") as f:
            return json.load(f)
    return loaders.load_schema()


def load_anchor_M(ds, wid):
    with open(os.path.join(config.HERE, ds["anchor_dir"], f"{wid:03d}.json"), encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    return sum(r["cost_mb"] for r in rows if r.get("cost_mb") is not None)


def load_explain_text(ds, wid, queries):
    blocks = []
    for q in queries:
        if ds["plan_has_wid"]:
            fp = os.path.join(config.HERE, ds["plan_dir"], f"{wid:03d}_{q['id']}.json")
        else:
            fp = os.path.join(config.HERE, ds["plan_dir"], f"{q['id']}.json")
        plan = None
        if os.path.exists(fp):
            with open(fp, encoding="utf-8") as f:
                plan = json.load(f).get("plan")
        if plan is not None:
            blocks.append(f"[{q['id']}]\n{render_plan(plan)}")
        else:
            blocks.append(f"[{q['id']}]\n(no plan tree)")
    return "\n\n".join(blocks)


def select_indexes(state, budget, max_attempts=4):
    state["budget"] = budget
    for attempt in range(max_attempts):
        try:
            rec = agents.run_selection(state)
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(1)
                continue
            return []
        if isinstance(rec, dict) and isinstance(rec.get("selected_indexes"), list):
            return [r for r in rec["selected_indexes"] if r.get("label") and r.get("props")]
    return []


def measure_all(session, queries, runs=5):
    out = {}
    for q in queries:
        server = [benchmark.run_timed(session, q["text"])["server_ms"] for _ in range(runs)]
        out[q["id"]] = round(benchmark.trimmed_mean(server), 3)
    return out


def _total_s(m):
    return sum(m.values()) / 1000.0


def _measure(session, queries, runs, before_s, by_key, picked):
    names, built_cost = [], 0.0
    for k in picked:
        c = by_key[k]
        name = "ma_" + c["label"].lower() + "_" + "_".join(p.lower() for p in c["props"])
        try:
            benchmark.create_index(session, name, c["label"], c["props"], c["on"])
            benchmark.wait_online(session, name)
            names.append(name)
            built_cost += c["cost_mb"] or 0.0
        except Exception:
            pass
    try:
        after = measure_all(session, queries, runs)
    finally:
        for name in names:
            try:
                benchmark.drop_index(session, name)
            except Exception:
                pass
    after_s = _total_s(after)
    speedup = (before_s - after_s) / before_s if before_s > 0 else 0.0
    return {"after_s": round(after_s, 4), "speedup": round(speedup, 6),
            "built_cost": round(built_cost, 4)}


def run_workload(ds, wid, runs, fracs, repeat):
    schema = load_schema_for(ds)
    schema_full = loaders.format_schema(schema, compact=False)
    schema_compact = loaders.format_schema(schema, compact=True)
    M = load_anchor_M(ds, wid)
    t_start = time.time()

    references, rag_s = [], 0.0
    state, orch_err, orch_s = None, None, 0.0
    orch_usage = {"model_wall_s": 0.0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    before_s = None

    driver = GraphDatabase.driver(config.NEO4J_URI, auth=config.NEO4J_AUTH,
                                  notifications_min_severity="OFF")
    try:
        with driver.session() as s:
            queries = loaders.load_workload(wid)
            workload_text = loaders._workload_text(queries)
            explain_text = load_explain_text(ds, wid, queries)

            if getattr(config, "RAG_ENABLED", False):
                _t = time.time()
                try:
                    references = rag.retrieve_references2(workload_text, k=config.RAG_TOP_K)
                except Exception:
                    references = []
                rag_s = round(time.time() - _t, 3)

            if ds["measure"]:
                for name in benchmark.show_user_indexes(s):
                    benchmark.drop_index(s, name)
                before_s = _total_s(measure_all(s, queries, runs))

            llm_client.reset_usage_stats()
            t0 = time.time()
            try:
                state, _ = orchestrator.run(
                    schema_stats=schema_full, schema_stats_compact=schema_compact,
                    workload_text=workload_text, explain_text=explain_text,
                    budget=M * 0.8, queries=queries, driver=driver,
                    logf=open(os.devnull, "w", encoding="utf-8"),
                    max_rounds=config.MAX_ROUNDS, run_selection=False,
                    references=references)
            except Exception as e:
                orch_err = str(e)[:300]
                state = None
            orch_s = time.time() - t0
            orch_usage = llm_client.get_usage_stats()

            if state is None:
                out = {"workload": wid, "M_mb": round(M, 4),
                       "rag_s": rag_s, "scoring_orch_wall_s": round(orch_s, 3),
                       "orchestrator_error": orch_err,
                       "wall_s": round(time.time() - t_start, 1), "results": []}
                if before_s is not None:
                    out["base_total_s"] = round(before_s, 4)
                return out

            by_key = {}
            for c in state["candidates"]:
                props = [str(p) for p in (c.get("props") or [])]
                label = (c.get("label") or "").strip()
                if not label or not props:
                    continue
                by_key[_key(c)] = {"label": label, "props": props, "on": c.get("on", "node"),
                                   "type": c.get("type", "range"),
                                   "cost_mb": c.get("cost_mb")}

            out = {"workload": wid, "M_mb": round(M, 4),
                   "candidates": len(state["candidates"]),
                   "rag_s": rag_s,
                   "scoring_model_s": round(orch_usage["model_wall_s"], 3),
                   "scoring_orch_wall_s": round(orch_s, 3),
                   "scoring_calls": orch_usage["calls"],
                   "scoring_prompt_tokens": orch_usage["prompt_tokens"],
                   "scoring_completion_tokens": orch_usage["completion_tokens"],
                   "orchestrator_error": orch_err,
                   "results": []}
            if before_s is not None:
                out["base_total_s"] = round(before_s, 4)

            meas_cache = {}
            for frac in fracs:
                pct = round(frac * 100)
                budget = M * frac
                for rep in range(repeat):
                    llm_client.reset_usage_stats()
                    t0 = time.time()
                    recs = select_indexes(state, budget)
                    rec_s = round(time.time() - t0, 3)
                    rec_usage = llm_client.get_usage_stats()
                    picked = []
                    cost = 0.0
                    for r in recs:
                        k = _key(r)
                        if k in by_key:
                            c = by_key[k]
                            picked.append(k)
                            cost += c["cost_mb"] or 0.0

                    entry = {
                        "budget_pct": pct, "budget_mb": round(budget, 4),
                        "actual_mb": round(cost, 4), "over_budget": cost > budget + 1e-9,
                        "indexes": picked,
                        "rec_model_s": rec_s,
                        "rec_prompt_tokens": rec_usage["prompt_tokens"],
                        "rec_completion_tokens": rec_usage["completion_tokens"],
                    }
                    if ds["measure"]:
                        key_tuple = tuple(sorted(picked))
                        if key_tuple not in meas_cache:
                            meas_cache[key_tuple] = _measure(s, queries, runs, before_s, by_key, picked)
                        m = meas_cache[key_tuple]
                        entry["before_s"] = round(before_s, 4)
                        entry["after_s"] = m["after_s"]
                        entry["speedup"] = m["speedup"]
                    if repeat > 1:
                        entry["rep"] = rep
                    out["results"].append(entry)

                    rep_tag = f"rep{rep} " if repeat > 1 else ""
                    msg = (f"  {pct:3d}% {rep_tag}({budget:8.4f}MB) -> {picked}  ={cost:.2f}MB"
                           + ("  [OVER BUDGET!]" if cost > budget + 1e-9 else "")
                           + f"  | select {rec_s:.1f}s/{rec_usage['prompt_tokens']}tok")
                    if ds["measure"]:
                        msg += f" speedup={m['speedup']:.3f}"
                    print(msg, flush=True)

            out["wall_s"] = round(time.time() - t_start, 1)
            return out
    finally:
        driver.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(config.DATASETS))
    ap.add_argument("workloads", nargs="*", type=int)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--fracs", type=str, default=None,
                    help="budget fractions, comma-separated (default: dataset-specific)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    ds = config.DATASETS[args.dataset]
    apply_dataset(ds)

    if args.fracs:
        fracs = tuple(float(x) for x in args.fracs.split(","))
    else:
        fracs = tuple(ds["fracs"])

    if args.out_dir:
        out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(config.HERE, args.out_dir)
    else:
        out_dir = os.path.join(config.HERE, ds["out_dir"])
    wids = args.workloads if args.workloads else resolve_workloads(ds)

    os.makedirs(out_dir, exist_ok=True)
    results = []
    for wid in wids:
        out_path = os.path.join(out_dir, f"{wid:03d}.json")
        if os.path.exists(out_path):
            print(f"[skip] workload {wid} already exists, skipping", flush=True)
            continue
        print(f"===== [{args.dataset}] workload {wid} =====", flush=True)
        out = run_workload(ds, wid, runs=args.runs, fracs=fracs, repeat=args.repeat)
        results.append(out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        if out.get("orchestrator_error"):
            print(f"  [scoring failed] {out['orchestrator_error'][:80]}", flush=True)
        else:
            print(f"  scoring {out['scoring_model_s']:.1f}s/{out['scoring_prompt_tokens']}tok "
                  f"(calls {out['scoring_calls']}, RAG {out['rag_s']:.1f}s, wall {out['wall_s']}s)\n",
                  flush=True)

    all_outs = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".json") and not fn.startswith("_"):
            with open(os.path.join(out_dir, fn), encoding="utf-8") as f:
                all_outs.append(json.load(f))
    all_outs.sort(key=lambda x: x.get("workload", 0))
    with open(os.path.join(out_dir, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_outs, f, ensure_ascii=False, indent=2)
    print(f"\nwritten to -> {out_dir}/  (total {len(all_outs)} workloads)")


if __name__ == "__main__":
    main()
