import llm_client
from loaders import format_single_prop_sets


def candidate_key(label, props, on="node"):
    if on == "relationship":
        return f"[:{label}]({','.join(props)})"
    return f"{label}({','.join(props)})"



def format_candidates(candidates, scores=None):
    if not candidates:
        return "(empty: no candidates generated yet)"
    lines = []
    for c in candidates:
        key = candidate_key(c["label"], c["props"], c.get("on", "node"))
        bits = [f"{key} | type={c.get('type', 'range')} | cost={c.get('cost_mb', '?')}MB"]
        if c.get("target_queries"):
            bits.append(f"target={','.join(c['target_queries'])}")
        if scores and key in scores:
            bits.append(f"score={scores[key].get('score', '?')}")
        lines.append("  " + " ".join(bits))
    return "\n".join(lines)


def format_plan_steps(plan_steps):
    if not plan_steps:
        return "(no history steps)"
    lines = []
    for st in plan_steps:
        lines.append(f"Round{st.get('round')} {st.get('action')}")
    return "\n".join(lines)


def _msg(system, user):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]



def judge_prompt(state):
    system = (
        "You are the routing/decision hub of a multi-agent system for property-graph database "
        "index recommendation. You do not build indexes or score anything yourself; you only decide "
        "which intermediate agent to call next.\n"
        "You can call three intermediate agents: generation (enumerate index candidates + "
        "estimate storage), scoring (give each candidate a 0-1 benefit score), selection (output "
        "the final index list in one shot), and Stop (end).\n"
        "Decision rule: the order is candidate generation → scoring → selection → Stop; "
        "selection must be called in one shot only after candidates have been generated AND all "
        "scored; never repeat an action that has already been completed."
        "You should also check whether the result of the previous step contains any obvious errors. If it contains obvious errors, rerun the previous step.\n"
    )
    user = (
        "Input\n"
        f"1. Database Schema (labels/relationships/properties, cardinality/selectivity, out-degree "
        f"distribution):\n{state['schema_stats_compact']}\n\n"
        f"2. Resource Budget (storage budget, MB): {state['budget']}\n\n"
        f"3. Candidate index set (with estimated storage and scored values; empty = not yet generated):\n"
        f"{format_candidates(state['candidates'], state['scores'])}\n\n"
        f"4. Previous Plan Steps (action history):\n{format_plan_steps(state['plan_steps'])}\n\n"
        "Task: judge which intermediate agent should be called next, output only one action name "
        "(no explanation):\n"
        "generation | scoring | selection | Stop"
    )
    return _msg(system, user)


def parse_action(text):
    t = (text or "").strip()
    if "stop" in t.lower():
        return "Stop"
    if "candidate" in t.lower() or "generate" in t.lower() or "generation" in t.lower():
        return "candidate_gen"
    if "scor" in t.lower():
        return "scoring"
    if "select" in t.lower():
        return "selection"
    return "Stop"


def run_judge(state):
    text = llm_client.call_llm(judge_prompt(state))
    return parse_action(text)



def generation_prompt(state):
    system = (
        "You are the index candidate generation agent. You analyze **query by query** each query in the "
        "workload and its physical execution plan tree, and find all index candidates (node indexes + "
        "relationship indexes) that can eliminate an expensive scan or sort.\n"
        "\n"
        "Core principle: a candidate must attack an expensive operator in the plan tree.\n"
        "An index candidate is valid only if it satisfies **both** of the following:\n"
        "  1. The property is a **filter key or sort key** in the query (equality/range/membership "
        "filter, or the sort key of ORDER BY … LIMIT, including an underlying property wrapped in a "
        "function);\n"
        "  2. It corresponds in the **physical execution plan tree** to an **expensive operator** — a "
        "NodeByLabelScan/AllNodesScan full scan at a leaf (large est), or a Sort/Top full sort.\n"
        "Build an index only for a filter/sort key that corresponds to an expensive scan/sort in the "
        "plan tree; both conditions are required.\n"
        "\n"
        "schema_stats field reference\n"
        "- Node label: `Label: N nodes` = this label has N nodes; each line under it is one property:\n"
        "    `prop: type null=X distinct=Y [min,max]`\n"
        "    · type       = property type (long/double/string/boolean)\n"
        "    · null=X     = fraction of nodes where this property is null (X=1 means all null, low index value)\n"
        "    · distinct=Y = fraction of distinct values = selectivity (Y=1.0 means unique per node, "
        "Y≈0 means very low cardinality). Low cardinality does NOT by itself make a range index "
        "low-value: if the property is the filter key of a whole-label aggregation (WHERE prop=$x "
        "RETURN count), the index still eliminates the entire label scan regardless of cardinality — "
        "judge by the scan's row count, not distinct.\n"
        "    · [min,max]  = value range of a numeric property\n"
        "    · multi flag = list-type property (one node can have multiple values)\n"
        "- Edge type: `TYPE: (src)->(dst), N edges, fanout[min,avg,max]`\n"
        "    · N edges  = total number of edges of this type\n"
        "    · fanout   = out-degree distribution of source nodes (min/average/max number of edges of "
        "this type leaving each source node);\n"
        "      large avg = traversal fan-out tends to explode, large max = super-nodes exist.\n"
        "\n"
        "Physical execution plan tree field reference\n"
        "One operator per line, indentation denotes parent-child. Format `operator [est=estimated rows] "
        "details`.\n"
        "- est = estimated rows of this operator, the larger the more expensive;\n"
        "- leaf NodeByLabelScan / AllNodesScan / relationship-type scan = full-scan starting point "
        "(expensive);\n"
        "- Filter's Details is the predicate (filter key); Expand(All) = expand neighbors along a "
        "relationship; Sort/Top = sort (expensive);\n"
        "- CacheProperties / Projection / ProduceResults = projection/caching/output, and do **not** "
        "represent filtering or sorting.\n"
        "\n"
        "Index types\n"
        "- Node index: FOR (n:Label) ON (n.prop1[, n.prop2...])\n"
        "- Relationship index: FOR ()-[r:TYPE]-() ON (r.prop)\n"
        "- Composite index: multiple properties on the same label/relationship type (Neo4j supports this).\n"
        "\n"
        "Single-property index set available in this database (complete list, from "
        "indexable_single_prop_sets.json)\n"
        "Each `Label(prop)` / `[:TYPE](prop)` below is a single-property index that can be built:\n"
        + format_single_prop_sets()
    )
    user = (
        "Analysis flow: first read the plan tree to find expensive operators, then cross-reference the "
        "queries to extract filter/sort keys, finally check against the list.\n"
        "\n"
        "Step 1: read each query's physical execution plan tree and mark expensive operators.\n"
        "For each query, find the full scans at the leaves (NodeByLabelScan/AllNodesScan) and Sort/Top, "
        "and note their est and the labels/properties involved.\n"
        "\n"
        "Step 2: go through the queries one by one and extract filter keys and sort keys.\n"
        "List every label.prop / [r:TYPE].prop that is used as a filter/sort key in each query "
        "(equality/range/membership filter, the sort key of ORDER BY … LIMIT, the underlying property "
        "wrapped in a function).\n"
        "Pure projection / pure caching columns (only RETURN / only CacheProperties, not involved in any "
        "filter or ORDER BY/LIMIT) must **not** be listed.\n"
        "\n"
        "Step 3: check against the list — build an index only for properties that are both a filter/sort "
        "key and correspond to an expensive operator from Step 1.\n"
        "For each key from Step 2, look back at Step 1: is its corresponding operator an expensive "
        "scan/sort?\n"
        "   - Yes (full scan with large est / full sort) → build the index (on=node uses Label, "
        "on=relationship uses TYPE);\n"
        "   - No (the query is near-zero cost, the scan est is tiny, or the key only appears in "
        "projection/caching) → do not build.\n"
        "When judging whether something is used as a filter/sort key, look only at the underlying "
        "property, regardless of how many layers of function/field extraction wrap it:\n"
        "   · anchor equality {prop: value} → unique/range index (scan → seek)\n"
        "   · equality/range/membership filter on label.prop inside WHERE/CASE (including prop IN list, "
        "and after collect(prop) then IN) → range/unique index\n"
        "   · property wrapped in a function for a derived-field comparison, e.g. "
        "datetime({epochMillis: x.birthday}).month/day → the underlying birthday is still the index key\n"
        "   · property used as a projection alias, but that alias drives a top-k sort via ORDER BY … LIMIT "
        "→ the underlying property is still the index key\n"
        "   · relationship property filtered/sorted (r.prop of [r:TYPE]) → relationship index\n"
        "   · multiple-property joint lookup on the same label/relationship type → composite index\n"
        "   · pure projection column (no filter, not involved in ORDER BY/LIMIT) → do not build an index\n"
        "\n"
        "Three hard prohibitions\n"
        "1. Forbidden to report candidates from memory such as 'LDBC/SNB common indexes' or 'often built "
        "on other workloads' — every candidate must point to which filter/sort key of which query of THIS "
        "workload it belongs to.\n"
        "2. Forbidden to build an index on a pure projection / pure caching property (only appears in "
        "RETURN / CacheProperties / ProduceResults, does not filter or sort).\n"
        "3. Forbidden to build an index on a key whose corresponding scan est in the plan tree is tiny, "
        "or whose query is near-zero cost — building it saves no time and only crowds the budget.\n"
        "\n"
        "Example\n"
        "Suppose the query:\n"
        "  MATCH (u:User {email: $email})-[:FOLLOWS]->(f:User)<-[:POSTED]-(m:Post)\n"
        "  WHERE m.publishedAt > $since\n"
        "  RETURN f.name, m.title ORDER BY m.publishedAt DESC LIMIT 20\n"
        "Plan-tree leaves: NodeByLabelScan [est=9000000] u:User (filtered by Filter u.email=$email), "
        "NodeByLabelScan [est=8000000] m:Post (filtered by Filter m.publishedAt>$since + Top sort).\n"
        "Candidates that can be built:\n"
        '  {"on":"node","label":"User","props":["email"],"type":"unique","target_queries":["q1"],'
        '"reason":"eliminates q1\'s User full scan (est=9M); email equality filter"}  # anchor equality\n'
        '  {"on":"node","label":"Post","props":["publishedAt"],"type":"range","target_queries":["q1"],'
        '"reason":"eliminates q1\'s Post full scan (est=8M) + full sort; publishedAt range filter + ORDER '
        'BY LIMIT"}  # range + sort\n'
        "f.name / m.title are only projections (ProduceResults/Projection), do not build an index on them.\n"
        "\n"
        "Four easy-to-miss pattern examples\n"
        "Example A function wrapping → the underlying property is still the filter key:\n"
        "  MATCH (m:Movie)\n"
        "  WHERE datetime({epochMillis: m.releasedAt}).year >= 2020\n"
        "  RETURN m\n"
        "  → m.releasedAt is wrapped in datetime({epochMillis: …}) and the derived .year is compared, but "
        "the underlying releasedAt is still the filter key → build Movie(releasedAt).\n"
        "Example B membership filter (after collect then IN) → still the filter key:\n"
        "  MATCH (t:Topic)<-[:TAGGED]-(:Product)\n"
        "  WITH collect(t.topicId) AS topics\n"
        "  MATCH (x:Topic)<-[:TAGGED]-(:Product)\n"
        "  WHERE x.topicId IN topics\n"
        "  RETURN x\n"
        "  → x.topicId appears in the IN membership test → build Topic(topicId); collect(…) only prepares "
        "the list for IN, it does not change that it is a filter key.\n"
        "Example C projection alias drives a top-k sort → the underlying property is still the sort key:\n"
        "  MATCH (e:Event)\n"
        "  RETURN e.createdAt AS c ORDER BY c DESC LIMIT 10\n"
        "  → although e.createdAt only appears via the alias c in ORDER BY, it drives the top-k sort → "
        "build Event(createdAt).\n"
        "Example D low-cardinality equality filter + count aggregation over a large label → still a "
        "must-build range index:\n"
        "  MATCH (n:LargeLabel)\n"
        "  WHERE n.attr = $x\n"
        "  RETURN count(n)\n"
        "  → the plan leaf is NodeByLabelScan(LargeLabel) with a large est (the whole label is scanned); "
        "the filter key is n.attr. Even if attr has very low cardinality (distinct≈0, only a handful of "
        "distinct values), the range index still eliminates that whole-label scan. Low cardinality does "
        "NOT disqualify the candidate — judge by the scan size it eliminates (the label's row count), not "
        "by the selectivity of attr.\n"
        "Example E union relationship types change only `target_queries` counting — NOT candidate generation:\n"
        "  MATCH (u:User)-[r:FOLLOWS|BLOCKS]->(v:User)\n"
        "  WHERE r.createdAt >= $since\n"
        "  RETURN count(*)\n"
        "  → r binds BOTH FOLLOWS and BLOCKS, so the filter r.createdAt>=... applies to both types. But each "
        "type is still one ordinary single-property relationship index, so generate ONE candidate per type as "
        "usual: [:FOLLOWS](createdAt) and [:BLOCKS](createdAt) — do NOT merge them and do NOT emit a duplicate "
        "object. The union only affects target_queries: list this query's id in target_queries of BOTH "
        "candidates, because the filter hits both types (not just the first type written in the union). When "
        "scoring each, use that type's own edge count, not the whole union est.\n"
        "\n"
        "Input\n"
        f"1. Database Schema:\n{state['schema_stats']}\n\n"
        f"2. The whole workload's query text:\n{state['workload_text']}\n\n"
        f"3. Each query's physical execution plan tree:\n{state['explain_text']}\n\n"
        "Output: output only one JSON object (no extra text):\n"
        "{\n"
        '  "candidates": [\n'
        '    { "on": "node"|"relationship", "label": "label or relationship type name", "props": ["p1","p2"],\n'
        '      "type": "unique|range|composite", "target_queries": ["q0422"],\n'
        '      "reason": "eliminates q0422\'s <Label> full scan (est=NNN); this query filters by p1 via '
        'equality/range, or ORDER BY LIMIT" }\n'
        "  ],\n"
        "}\n"
        "- \"on\" must declare whether this candidate is a node (\"node\") or relationship (\"relationship\") "
        "index;\n"
        "- \"label\" is the node label name (on=node) or relationship type name (on=relationship);\n"
        "- \"reason\" is required: state which filter/sort key of which query this candidate corresponds to, "
        "and which expensive operator (scan/sort) in the plan tree it eliminates, plus its est;\n"
        "- \"target_queries\" lists the query ids it serves (e.g. q0422);\n"
        "- You do not need to estimate cost_mb (storage); the system will fill it in automatically with the "
        "estimate_index_storage tool."
    )
    return _msg(system, user)


def run_candidate_gen(state):
    data = llm_client.call_llm_json(generation_prompt(state))
    if isinstance(data, list):
        data = {"candidates": data}
    candidates = data.get("candidates", [])
    for c in candidates:
        c.setdefault("on", "node")
        if c.get("on") not in ("node", "relationship"):
            c["on"] = "node"
        c.setdefault("type", "range")
        c.setdefault("props", [])
        c.setdefault("target_queries", [])
    return {"candidates": candidates}



def scoring_prompt(state):
    system = (
        "You are the index scoring agent. The candidate index set is fixed; now you score every candidate "
        "in one shot (a 0-1 benefit score).\n"
        "The only scoring criterion is 'how much this index, built alone, reduces the total execution time "
        "of the whole workload'. You must combine the schema, the query text, and the physical execution "
        "plan tree to judge the real time saved yourself, rather than only looking at distinct/cardinality.\n"
        "\n"
        "Step 0 (mandatory, before scoring): first circle the 'full set of queries served' by each candidate correctly.\n"
        "The schema gives parent/child label relationships: the node count of a parent label = sum of the "
        "node counts of its child labels, and a child-label node also carries the parent label.\n"
        "But Neo4j indexes are built per label: a parent-label index only serves queries that bind the "
        "parent label, and does **not** serve queries that bind each child label\n"
        "(e.g. a Message(creationDate) index only serves queries binding :Message, not queries binding "
        ":Post/:Comment; the latter need their own Post(creationDate)/Comment(creationDate) indexes).\n"
        "When circling 'serves' for each candidate, only circle queries that bind that label; do not circle "
        "child-label queries into a parent-label index, otherwise you will severely overestimate the parent "
        "index's value as 'covering all child labels' — this is the most fatal scoring error.\n"
        "\n"
        "Core of judging real time saved: can the index eliminate an 'expensive scan' or an 'expensive sort'?\n"
        "1. The value of a range/sort index (range + ORDER BY … LIMIT) = the 'scan + sort' cost it "
        "eliminates; look at **two** places in the plan tree:\n"
        "   (a) where the leaf is: leaf is NodeByLabelScan(that label) with large est → eliminates the whole "
        "scan, high value; leaf is Expand (a small set traversed from an anchor) → only eliminates a "
        "small-set sort, low value.\n"
        "   (b) the sort operator type: for ORDER BY … LIMIT, does the plan tree use 'Sort + Limit' (fully "
        "sorting all filtered rows) or 'Top' (a heap keeping only the top K)? This decides why the same "
        "ORDER BY LIMIT can differ several-fold in value:\n"
        "       Sort + Limit = without an index it must fully sort O(n log n); the index eliminates the "
        "'full sort' too, value is **highest**;\n"
        "       Top = without an index it only does top-K O(n log K) (already cheap); the index only saves "
        "the scan, value is **one tier lower**.\n"
        "   Look at both: first the leaf (full scan or small set), then the sort operator (Sort or Top). "
        "Leaf full scan + Sort = highest value; leaf small set, or sort is Top = value clearly drops.\n"
        "2. The value of an anchor equality lookup ({prop: $param}) = the scan cost it eliminates × the "
        "number of queries it serves; look at the plan tree + serves:\n"
        "   Look at how large the NodeByLabelScan est is under Filter(prop=$param), and how many queries "
        "this index serves (the serves count):\n"
        "   large est (millions) + serves many queries → high score (each eliminates one full scan, large "
        "cumulative saving);\n"
        "   large est but serves only 1 query → the total saving is only that one scan, score clearly lower "
        "(don't give it the same tier as an index serving many queries);\n"
        "   small est (small table), or the plan tree already has a seek / no full scan → low score.\n"
        "   ⚠️ 'returns only 1 row' does NOT equal 'near-zero cost' — without an index, finding that 1 row "
        "may require a full scan of millions of rows; that scan is what the index saves.\n"
        "3. Reverse-traversal key: when forward traversal from an anchor would fan out explosively, "
        "traversing backwards from this property's starting point yields very high benefit.\n"
        "\n"
        "Scores must spread into distinct tiers: score by 'leaf scan size + sort operator type', don't "
        "crowd everything into 0.7~0.9.\n"
        "- Leaf NodeByLabelScan with est in the millions + sort operator Sort (full sort) → 0.9~1.0 (the "
        "only highest-tier jackpot);\n"
        "- Leaf NodeByLabelScan with est in the millions + sort operator Top (top-K) → 0.3~0.5 (Top itself "
        "is cheap, clearly one tier lower);\n"
        "- Leaf NodeByLabelScan but est very small (small table, <100k rows) → 0.05~0.2;\n"
        "- Leaf is Expand (a small set traversed from an anchor) → 0.1~0.2.\n"
        "The tiers above describe how much a 'single query' saves (leaf scan size + sort operator type); "
        "final score = single-query saving × the number of queries served (serves). "
        "Look at and multiply both: a small-table scan serving many queries still has small per-query "
        "saving, so its total cannot rise; a large-table full scan / full sort saving a lot even if it "
        "serves only 1 query. "
        "Don't stare only at 'serves count' or only at 'est size'; their product is this index's total "
        "saving across the whole workload.\n"
        "\n"
        "Three low-value / harmful classes, give a low score 0~0.1.\n"
        "a. An index on a small label: very few nodes (the label's scan est in the plan tree is tiny), "
        "index saving ≈ 0.\n"
        "b. A range filter that only does leaf filtering, eliminating neither the start-point scan nor the "
        "sort (e.g. a range filter on a relationship property, or a range condition on a passing node that "
        "is neither an anchor nor involved in ORDER BY LIMIT).\n"
        "c. The key is already a seek / index lookup in the plan tree (not a full scan) — building another "
        "index saves no scan, saving ≈ 0.\n"
        "\n"
        "Four common mistakes, be sure to avoid them.\n"
        "- Mistake 1: seeing an 'equality point lookup (distinct=1.0)' and only checking whether its "
        "eliminated single scan est is large, ignoring how many queries it serves. Wrong — the correct "
        "approach is 'single scan cost × number of queries served': est in the millions and serving many "
        "queries → high score (large cumulative saving); est in the millions but serving only 1 query → it "
        "only saves one scan of that one query, total saving limited, score clearly lower; est tiny / "
        "already a seek → low score. ⚠️ 'returns only 1 row' does not equal 'near-zero cost' — without an "
        "index, finding that 1 row may require a full scan of millions of rows, and that scan is what the "
        "index saves; but 'serves only 1 query' means the total saving is only one copy, so the score "
        "cannot be high.\n"
        "- Mistake 2: seeing a 'range filter' and applying 'lookup-back is more expensive' to give a low "
        "score. Wrong — 'lookup-back is more expensive' only applies to the case where 'the filter hits "
        "many rows AND must return those rows AND there is no ORDER BY … LIMIT short-circuit'; once the "
        "query has ORDER BY … LIMIT, a range index is the highest value, and must never be downgraded "
        "because of 'lookup-back'.\n"
        "- Mistake 3: seeing 'ORDER BY … LIMIT' and not looking at the leaf, defaulting that a range index "
        "beats every point-lookup index. Wrong — a range index counts as 'eliminating full scan + full "
        "sort' only when the leaf is NodeByLabelScan(that label); if the leaf is Expand (a small set "
        "traversed from an anchor), it only saves a small-set sort. In that case a point-lookup id index "
        "that 'serves many short queries, each eliminating that label's full scan' has higher value "
        "instead. Look at the plan-tree leaf + the served query count for both, then decide high/low.\n"
        "- Mistake 4: seeing 'parent-label node count = sum of child-label node counts' and circling "
        "child-label queries into a parent-label index. Wrong — parent/child label relationship only means "
        "they have a hierarchy (a child-type node also carries the parent-type label), but Neo4j indexes "
        "are built per label; a parent-label index only serves queries binding the parent label, not "
        "queries binding a child label (child labels need their own indexes). When circling 'serves' for an "
        "index, only count queries that bind that label, and don't count child-label queries into a "
        "parent-label index.\n"
        "\n"
        "Global view: the score targets the total saving across the whole workload, not a single query."
    )
    user = (
        "Input\n"
        f"1. The whole workload's query text:\n{state['workload_text']}\n\n"
        f"2. Database Schema (cardinality/selectivity/out-degree distribution):\n{state['schema_stats']}\n\n"
        f"3. Each query's physical execution plan tree (operator [est=estimated rows] details; leaf scan = "
        f"full-scan starting point):\n{state['explain_text']}\n\n"
        f"4. Candidate index set (with estimated storage):\n{format_candidates(state['candidates'])}\n\n"
        f"5. Reference experience (index-recommendation experience from others, heuristic guidance "
        f"only; the real value of this workload is still determined by the query/plan tree in sections 1-4, "
        f"and you must not copy concrete index names from the experience):\n"
        f"{state.get('experience_text') or '(none)'}\n\n"
        "Task: give each candidate a 0-1 score s(i); output only one JSON object (key = 'Label(property)', "
        "composite index uses 'Label(property1,property2)', relationship index uses '[:RelationshipType]"
        "(property)'):\n"
        "{\n"
        '  "Message(creationDate)": { "serves": ["q0002"], "score": 0.9, "reason": "range filter + ORDER BY '
        'LIMIT sort key; range scan short-circuits, eliminating full scan + full sort" },\n'
        '  "Message(id)": { "serves": ["q0001"], "score": 0.9, "reason": "point lookup Filter(id=$param) sits '
        'over NodeByLabelScan(Message) est≈3M; the index turns full scan into seek, saving the scan" }\n'
        "}\n"
        "- 'serves' lists all query ids this index serves: only count queries that bind that label (Neo4j "
        "indexes are per label; a parent-label index does not serve child-label queries, and child-label "
        "queries must be circled into the child label's own index) — be sure to circle correctly;\n"
        "- score reflects the relative size of 'how much execution time this index, built alone, saves'; "
        "the one saving the most time gets the highest score;\n"
        "- give a short reason for each, stating the basis for how much time you judge it saves (whether it "
        "eliminates a scan or a sort)."
    )
    return _msg(system, user)


def run_scoring(state):
    return llm_client.call_llm_json(scoring_prompt(state))



def selection_prompt(state):
    system = (
        "You are the selection agent. All candidates are generated and scored; now you output "
        "the final selection in one shot.\n"
        "Your core goal: within the storage budget, choose the most beneficial candidate index set.\n"
        "Combine the query text, the physical execution plan tree, and the schema to select. consider the interactions among the individual queries.\n"
        "Second priority — cost-effectiveness (saving ÷ storage).\n"
        "When two candidates have similar saving, prefer the smaller-storage one; the tighter the budget, "
        "the more cost-effectiveness matters.\n"
        "\n"
        "Budget discipline\n"
        "Hard constraint: your selected_indexes MUST NOT exceed the storage budget in section 5"
    )
    user = (
        "Input\n"
        f"1. Candidate index set + score s(i) + estimated storage cost(i) + target queries:\n"
        f"{format_candidates(state['candidates'], state['scores'])}\n\n"
        f"2. The whole workload's query text:\n{state['workload_text']}\n\n"
        f"3. Each query's physical execution plan tree (operator [est=estimated rows] details; leaf scan = "
        f"full-scan starting point):\n{state['explain_text']}\n\n"
        f"4. Database Schema (node count / selectivity distinct / out-degree fanout):\n"
        f"{state['schema_stats']}\n\n"
        f"5. Resource Budget (storage budget, MB): {state['budget']}\n\n"
        "Task: output only one JSON object:\n"
        "{\n"
        '  "selected_indexes": [ { "on":"node", "label":"Person", "props":["id"], "type":"unique", '
        '"score":0.95, "cost_mb":0.26 } ],\n'
        "}\n"
        "cost_mb directly from the input, don't fabricate them;\n"
        "- If no candidate has significant saving, selected_indexes outputs an empty array []."
    )
    return _msg(system, user)


def run_selection(state):
    return llm_client.call_llm_json(selection_prompt(state))



_REF_QUERY_CAP = 3000
_REF_PLAN_CAP = 6000
_REF_HEURISTIC_CAP = 1200


def _truncate_ref(text):
    pm = "Execution plan tree"
    bm = "heuristic optimal solution"
    ip = text.find(pm)
    ib = text.find(bm)
    if ip == -1 and ib == -1:
        cap = _REF_QUERY_CAP + _REF_PLAN_CAP
        return text[:cap] + (" ...(truncated)" if len(text) > cap else "")

    if ip == -1:
        head, plan, tail = text[:ib], "", text[ib:]
    elif ib == -1:
        head, plan, tail = text[:ip], text[ip:], ""
    else:
        head, plan, tail = text[:ip], text[ip:ib], text[ib:]

    def cut(s, n):
        return s if len(s) <= n else s[:n] + " ...(truncated)"

    return cut(head, _REF_QUERY_CAP) + cut(plan, _REF_PLAN_CAP) + cut(tail, _REF_HEURISTIC_CAP)


def _format_reference_blocks(references):
    lines = []
    for r in references:
        text = r.get("text", "")
        lines.append(f"### {r['ref_id']} source={r['source']} | {r['heading']} ({r['kind']})\n{_truncate_ref(text)}")
    return "\n\n".join(lines)


def assistant_prompt(state, references):
    system = (
        "You are the reference-material assistant agent. The scoring agent will receive a batch of reference "
        "materials (experience optimal solutions of heuristics "
        "workloads).\n"
        "Your job: judge one by one whether each piece of reference material is useful for **scoring the "
        "current workload**, keeping only the material that is 'genuinely relevant, transferable, and won't "
        "mislead', and removing the material that is 'unrelated to the current workload's query patterns / "
        "plan-tree features and would only mislead the scoring'.\n"
        "\n"
        "Judging criterion: whether the query patterns and index trade-offs described by the material "
        "(point lookup / range+ORDER BY LIMIT / relationship-property filter / parent label does not serve "
        "child label / unique ID point lookup / cost ladder, etc.) actually appear in the current "
        "workload's query text or physical execution plan tree.\n"
        "- Useful (keep): the pattern matches and can inspire the scoring;\n"
        "- Useless (remove): the pattern does not match (e.g. the current workload is all point lookups but "
        "the material talks about range sorting; or the concrete labels/properties in the material don't "
        "exist in the current schema and can't be analogized).\n"
        "Prefer keeping a small amount of highly-relevant material over stuffing in irrelevant material "
        "that misleads scoring. If all are irrelevant, keep the empty set."
    )
    user = (
        "Current workload context\n"
        f"1. Query text:\n{state['workload_text']}\n\n"
        f"2. Database Schema:\n{state['schema_stats']}\n\n"
        f"3. Physical execution plan tree:\n{state['explain_text']}\n\n"
        "Candidate reference material\n"
        + _format_reference_blocks(references)
        + "\n\n"
        "Task: judge one by one whether each piece of reference material is useful; output only one JSON "
        "object:\n"
        '{"keep": [1, 3], "reason": "brief reason for keeping/removing"}\n'
        "- The numbers in the keep array = the ids of the reference material you consider 'useful and worth "
        "keeping' (corresponding to R1/R2/... above);\n"
        "- If all are useless, keep outputs an empty array [];\n"
        "- Output only JSON, no extra text."
    )
    return _msg(system, user)


def run_assistant(state):
    references = state.get("references") or []
    if not references:
        return []
    try:
        data = llm_client.call_llm_json(assistant_prompt(state, references))
        keep = data.get("keep") if isinstance(data, dict) else None
        if not isinstance(keep, list):
            return references
        by_id = {r["ref_id"]: r for r in references}
        kept = []
        for x in keep:
            if isinstance(x, (int, float)):
                rid = f"R{int(x)}"
            else:
                rid = str(x).strip()
            if rid in by_id:
                kept.append(by_id[rid])
        return kept
    except Exception:
        return references


