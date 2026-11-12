import os
import config


def load_schema():
    import json
    with open(config.SCHEMA_JSON, encoding="utf-8") as f:
        return json.load(f)


def _fmt_prop(name, p):
    parts = [f"{name}:{p.get('type', '?')}"]
    if p.get("is_multi_valued"):
        parts.append("multi")
    if "null_ratio" in p:
        parts.append(f"null={p['null_ratio']:.3f}")
    if "distinct_ratio" in p:
        parts.append(f"distinct={p['distinct_ratio']:.3f}")
    if "true_ratio" in p:
        parts.append(f"true={p['true_ratio']:.4f}")
    if "val_min" in p and "val_max" in p:
        parts.append(f"[{p['val_min']},{p['val_max']}]")
    return " ".join(parts)


def _find_label_hierarchy(labels):
    import itertools
    counts = {}
    for name, info in labels.items():
        if isinstance(info, dict):
            counts[name] = info.get("node_count", 0)
        else:
            counts[name] = info
    parents = []
    for name, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        others = [n for n, c in counts.items() if n != name and 0 < c < cnt]
        found = None
        for r in range(2, min(5, len(others) + 1)):
            for combo in itertools.combinations(others, r):
                if sum(counts[n] for n in combo) == cnt:
                    found = sorted(combo, key=lambda n: -counts[n])
                    break
            if found:
                break
        if found:
            parents.append((name, found))
    return parents


def format_schema(schema, compact=False):
    out = []
    labels = schema.get("labels", {})
    out.append(f"=== Node labels (total {len(labels)}) ===")
    for lbl, info in labels.items():
        n = info.get("node_count", 0)
        props = info.get("properties", {})
        out.append(f"{lbl}: {n} nodes")
        for pname, p in props.items():
            if compact:
                out.append(f"  {pname}: distinct={p.get('distinct_ratio', '?')}")
            else:
                out.append(f"  {_fmt_prop(pname, p)}")

    parents = _find_label_hierarchy(labels)
    if parents:
        out.append("\n=== Parent/child label relationships (parent node count = sum of child node counts; "
                   "child-label nodes also carry the parent label; note Neo4j indexes are per label, and a "
                   "parent-label index only serves queries that bind the parent label) ===")
        for parent, children in parents:
            out.append(f"{parent} is the parent label of {', '.join(children)}")

    edges = schema.get("edge_types", {})
    out.append(f"\n=== Edge types (total {len(edges)}) ===")
    for etype, info in edges.items():
        src = ",".join(info.get("src_labels", []))
        dst = ",".join(info.get("dst_labels", []))
        out.append(
            f"{etype}: ({src})->({dst}), {info.get('edge_count', 0)} edges, "
            f"fanout[min={info.get('fanout_min', '?')}, avg={info.get('fanout_avg', '?')}, "
            f"max={info.get('fanout_max', '?')}]"
        )
        for pname, p in (info.get("properties") or {}).items():
            if compact:
                out.append(f"    {pname}: distinct={p.get('distinct_ratio', '?')}")
            else:
                out.append(f"    {_fmt_prop(pname, p)}")

    idxs = schema.get("existing_indexes", [])
    if idxs:
        out.append(f"\n=== Existing indexes ({len(idxs)}) ===")
        for i in idxs:
            out.append(
                f"  {i.get('type')} on {i.get('labelsOrTypes')} {i.get('properties')} "
                f"[{i.get('state')}]"
            )
    else:
        out.append("\n=== Existing indexes ===\n  (none, only built-in LOOKUP)")
    return "\n".join(out)


def load_single_prop_sets():
    import json
    with open(config.SINGLE_PROP_SETS_JSON, encoding="utf-8") as f:
        return json.load(f)


def format_single_prop_sets(sets=None):
    if sets is None:
        sets = load_single_prop_sets()
    lines = ["Node single-property indexes:"]
    for label, props in sets.get("nodes", {}).items():
        lines.append(f"  {label}: {', '.join(props)}")
    lines.append("Relationship single-property indexes (only the listed edge types have properties; other edge types have no properties and cannot have a single-property relationship index):")
    rels = [f"[:{t}]({', '.join(props)})" for t, props in sets.get("relationships", {}).items()]
    lines.append("  " + ", ".join(rels))
    return "\n".join(lines)


def load_workload(workload_id):
    d = os.path.join(config.DATASET_DIR, f"workload_{workload_id}")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"workload directory not found: {d}")
    queries = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".cypher"):
            continue
        with open(os.path.join(d, fn), encoding="utf-8") as f:
            queries.append({"id": fn[:-7], "text": f.read().strip()})
    return queries


def _workload_text(queries):
    blocks = []
    for q in queries:
        blocks.append(f"### {q['id']}\n{q['text']}")
    return "\n\n".join(blocks)
