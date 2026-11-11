import argparse
import math

from neo4j import GraphDatabase

URI = "your_neo4j_uri" 
AUTH = ("your_username", "your_password")  

PAGE_SIZE = 8192            
VALUE_BYTES = 8             
ENTRY_HEADER = 5.8          
FILL_FACTOR = 1.0           
INTERNAL_FACTOR = 1.0 # adjust this factor before running   
PAGE_FIXED_BYTES = 32 * 1024  
MB = 1024 * 1024


def _btick(name):
    return "`" + str(name).replace("`", "") + "`"


def _detect_type(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "long"
    if isinstance(v, float):
        return "double"
    if isinstance(v, str):
        return "string"
    if isinstance(v, (list, tuple, dict)):
        return "list"
    return "other"


def _head(kind, label):
    if kind == "relationship":
        return f"()-[r:{_btick(label)}]->()", "r"
    return f"(n:{_btick(label)})", "n"


def _key_bytes(session, label, prop, kind="node"):
    head, var = _head(kind, label)
    v = session.run(
        f"MATCH {head} WHERE {var}.{_btick(prop)} IS NOT NULL "
        f"RETURN {var}.{_btick(prop)} AS v LIMIT 1"
    ).single()
    if v is None:
        return "none", 0.0
    t = _detect_type(v["v"])
    if t in ("long", "double"):
        return t, 8.0
    if t == "bool":
        return t, 1.0
    if t == "string":
        rec = session.run(
            f"MATCH {head} WHERE {var}.{_btick(prop)} IS NOT NULL "
            f"RETURN avg(size({var}.{_btick(prop)})) AS a"
        ).single()
        return t, float(rec["a"] or 0.0)
    return t, 8.0


def _n_nonnull(session, label, props, kind="node"):
    head, var = _head(kind, label)
    if len(props) == 1:
        p = _btick(props[0])
        rec = session.run(
            f"MATCH {head} RETURN count({var}.{p}) AS c"
        ).single()
        return int(rec["c"] or 0)
    conds = " AND ".join(f"{var}.{_btick(p)} IS NOT NULL" for p in props)
    rec = session.run(
        f"MATCH {head} WHERE {conds} RETURN count({var}) AS c"
    ).single()
    return int(rec["c"] or 0)


def pages_from(n_nonnull, key_bytes, internal_factor=INTERNAL_FACTOR):
    entry_bytes = key_bytes + VALUE_BYTES + ENTRY_HEADER
    entries_per_page = math.floor(PAGE_SIZE * FILL_FACTOR / entry_bytes) if entry_bytes > 0 else 0
    leaf_pages = math.ceil(n_nonnull / entries_per_page) if entries_per_page > 0 else 0
    total_pages = math.ceil(leaf_pages * internal_factor)
    total_bytes = total_pages * PAGE_SIZE + PAGE_FIXED_BYTES
    return leaf_pages, total_pages, total_bytes


def _estimate(label, props, driver, kind="node", internal_factor=INTERNAL_FACTOR):
    with driver.session() as s:
        types = []
        total_key_bytes = 0.0
        for p in props:
            t, kb = _key_bytes(s, label, p, kind)
            types.append((p, t))
            total_key_bytes += kb
        n_nonnull = _n_nonnull(s, label, props, kind)

    entry_bytes = total_key_bytes + VALUE_BYTES + ENTRY_HEADER
    entries_per_page = math.floor(PAGE_SIZE * FILL_FACTOR / entry_bytes) if entry_bytes > 0 else 0
    leaf_pages = math.ceil(n_nonnull / entries_per_page) if entries_per_page > 0 else 0
    total_pages = math.ceil(leaf_pages * internal_factor)
    total_bytes = total_pages * PAGE_SIZE + PAGE_FIXED_BYTES
    return {
        "label": label,
        "kind": kind,
        "props": props,
        "types": types,
        "n_nonnull": n_nonnull,
        "total_key_bytes": round(total_key_bytes, 3),
        "entry_bytes": round(entry_bytes, 3),
        "entries_per_page": entries_per_page,
        "leaf_pages": leaf_pages,
        "total_pages": total_pages,
        "page_size": PAGE_SIZE,
        "fill_factor": FILL_FACTOR,
        "internal_factor": internal_factor,
        "page_fixed_bytes": PAGE_FIXED_BYTES,
        "total_bytes": total_bytes,
        "mb": round(total_bytes / MB, 4),
    }


def estimate_index_storage_detail(label, props, driver=None, kind="node",
                                  internal_factor=INTERNAL_FACTOR):
    own = driver is None
    if own:
        driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        return _estimate(label, list(props), driver, kind, internal_factor)
    finally:
        if own:
            driver.close()


def estimate_index_storage(label, props, driver=None, kind="node",
                           internal_factor=INTERNAL_FACTOR):
    return estimate_index_storage_detail(label, list(props), driver, kind,
                                         internal_factor)["mb"]


def main():
    ap = argparse.ArgumentParser(description="估算 Neo4j 索引存储占用（MB，v2：O=13.8 + 40KB 地板）")
    ap.add_argument("label")
    ap.add_argument("props", nargs="+")
    ap.add_argument("--uri", default=URI)
    ap.add_argument("--user", default=AUTH[0])
    ap.add_argument("--password", default=AUTH[1])
    args = ap.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    d = estimate_index_storage_detail(args.label, args.props, driver)
    driver.close()

    print(f"candidate indexs: {d['label']}({', '.join(d['props'])})")
    print(f"  type: {[(p, t) for p, t in d['types']]}")
    print(f"  non-null nodes: {d['n_nonnull']:,}")
    print(f"  per entry = key {d['total_key_bytes']} + id {VALUE_BYTES} + header {ENTRY_HEADER} = {d['entry_bytes']} B")
    print(f"  per page {d['entries_per_page']} entries -> leaf {d['leaf_pages']} pages -> total {d['total_pages']} pages × {PAGE_SIZE} B")
    print(f"  fixed items +{PAGE_FIXED_BYTES} B")
    print(f"  estimated storage: {d['mb']} MB")


if __name__ == "__main__":
    main()
