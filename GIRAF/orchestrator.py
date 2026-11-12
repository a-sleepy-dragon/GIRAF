import agents
import config
import estimate
import rag

MAX_CAND_GEN = 3
MAX_SCORE = 3


def _log(msg, logf):
    line = str(msg)
    print(line, flush=True)
    logf.write(line + "\n")
    logf.flush()


def fill_cost_mb(candidates, driver):
    for c in candidates:
        label = c["label"]
        props = c["props"]
        if not props:
            c["cost_mb"] = None
            continue
        try:
            c["cost_mb"] = round(
                estimate.estimate_index_storage(label, props, driver, on=c.get("on", "node")), 4)
        except Exception as e:
            c["cost_mb"] = None
            c["cost_err"] = str(e)[:120]


def _make_state(schema_stats, schema_stats_compact, workload_text, explain_text,
                budget, queries, references=None):
    return {
        "schema_stats": schema_stats,
        "schema_stats_compact": schema_stats_compact,
        "workload_text": workload_text,
        "explain_text": explain_text,
        "budget": budget,
        "queries": queries,
        "references": references or [],
        "experience_text": "",
        "candidates": [],
        "scores": {},
        "selection": None,
        "plan_steps": [],
    }


def run(schema_stats, schema_stats_compact, workload_text, explain_text,
        budget, queries, driver, logf, max_rounds=None, run_selection=True,
        references=None):
    max_rounds = max_rounds or config.MAX_ROUNDS
    state = _make_state(schema_stats, schema_stats_compact, workload_text,
                        explain_text, budget, queries, references)
    if state["references"]:
        try:
            if getattr(config, "RAG_ASSISTANT_ENABLED", True):
                kept = agents.run_assistant(state)
                state["experience_text"] = rag.format_references(kept)
                kept_desc = ", ".join(f"{r['ref_id']}:{r['source']}|{r['heading']}" for r in kept) or "(none)"
                _log(f"[assistant] {len(state['references'])} reference item(s) -> kept {len(kept)} [{kept_desc}]", logf)
            else:
                state["experience_text"] = rag.format_references(state["references"])
                _log(f"[assistant] disabled: include all {len(state['references'])} reference items", logf)
        except Exception as e:
            state["experience_text"] = rag.format_references(state["references"])
            _log(f"[assistant] failed, fallback to include all: {str(e)[:160]}", logf)
    _log(f"[init] workload has {len(queries)} queries, budget {budget} MB, at most {max_rounds} rounds", logf)

    force_next = None
    last_cand_keys = None
    no_change = 0
    last_rec_keys = None
    rec_no_change = 0
    cand_gen_count = 0
    score_count = 0

    for rnd in range(1, max_rounds + 1):
        if force_next:
            action = force_next
            force_next = None
        else:
            action = agents.run_judge(state)
        _log(f"[Round {rnd}] router -> {action}", logf)

        if action == "Stop" and state["candidates"] and state["selection"] is None:
            if not run_selection:
                if not state["scores"]:
                    action = "scoring"
                    _log("  (candidates exist but not scored before Stop, force scoring first)", logf)
            elif not state["scores"]:
                action = "scoring"
                _log("  (candidates exist but not scored before Stop, force scoring first)", logf)
            else:
                action = "selection"
                _log("  (candidates + scores exist but not selected before Stop, force selection first)", logf)

        if not run_selection and action == "selection":
            if not state["scores"]:
                _log("  (selection needs scores, score first)", logf)
                action = "scoring"
            else:
                _log("  (skip ordering selection: selection is handled by the eval layer per budget tier)", logf)
                action = "Stop"

        if action == "Stop":
            state["plan_steps"].append({"round": rnd, "action": "Stop"})
            _log("[Stop] router decided to end", logf)
            break

        if action == "candidate_gen" and cand_gen_count >= MAX_CAND_GEN:
            action = "selection" if state["scores"] else "scoring"
            _log(f"  (candidate generation already ran {cand_gen_count} rounds without converging, force {action})", logf)
        elif action == "scoring" and score_count >= MAX_SCORE:
            action = "selection"
            _log(f"  (scoring already ran {score_count} rounds without converging, force selection)", logf)

        if action == "scoring" and not state["candidates"]:
            _log("  (no candidates, force candidate generation first)", logf)
            action = "candidate_gen"
        elif action == "selection" and not state["candidates"]:
            _log("  (no candidates, force candidate generation first)", logf)
            action = "candidate_gen"
        elif action == "selection" and not state["scores"]:
            _log("  (not scored yet, force scoring first)", logf)
            action = "scoring"

        try:
            if action == "candidate_gen":
                cand_gen_count += 1
                out = agents.run_candidate_gen(state)
                fill_cost_mb(out["candidates"], driver)
                state["candidates"] = out["candidates"]
                _log(f"  candidate generation: {len(out['candidates'])} candidates", logf)
                for c in out["candidates"]:
                    _log(f"    - {agents.candidate_key(c['label'], c['props'], c.get('on', 'node'))} "
                         f"type={c['type']} cost={c['cost_mb']}MB", logf)
                new_keys = {agents.candidate_key(c["label"], c["props"], c.get("on", "node"))
                            for c in out["candidates"]}
                if last_cand_keys is not None and new_keys == last_cand_keys:
                    no_change += 1
                else:
                    no_change = 0
                last_cand_keys = new_keys
                if no_change >= 1:
                    if state["scores"]:
                        force_next = "selection"
                        _log("  (candidate set and scores both stable, force selection next round to avoid a candidate<->scoring loop)", logf)
                    else:
                        force_next = "scoring"
                        _log("  (candidate set identical to last time, force scoring next round to avoid a loop)", logf)

            elif action == "scoring":
                score_count += 1
                state["scores"] = agents.run_scoring(state)
                _log(f"  scoring: {len(state['scores'])} candidates scored", logf)

            elif action == "selection":
                state["selection"] = agents.run_selection(state)
                _log(f"  selection: {len(state['selection'].get('selected_indexes', []))} indexes", logf)
                rec_keys = tuple(sorted(
                    agents.candidate_key(c["label"], c["props"], c.get("on", "node"))
                    for c in state["selection"].get("selected_indexes", [])
                    if c.get("label") and c.get("props")
                ))
                if last_rec_keys is not None and rec_keys == last_rec_keys:
                    rec_no_change += 1
                else:
                    rec_no_change = 0
                last_rec_keys = rec_keys
                if rec_no_change >= 1:
                    force_next = "Stop"
                    _log("  (selection identical to last time, force end next round to avoid a loop)", logf)

        except Exception as e:
            _log(f"  !! {action} failed: {str(e)[:300]}", logf)
            state["plan_steps"].append({
                "round": rnd, "action": action, "error": str(e)[:200]
            })
            continue

        state["plan_steps"].append({"round": rnd, "action": action})

    return state, _assemble_report(state)


def _assemble_report(state):
    scores_plain = {}
    for k, v in state["scores"].items():
        scores_plain[k] = v.get("score", v) if isinstance(v, dict) else v
    rec = state["selection"] or {}
    return {
        "schema_stats": state["schema_stats_compact"],
        "budget": state["budget"],
        "workload": [q["id"] for q in state["queries"]],
        "scores": scores_plain,
        "selected_indexes": rec.get("selected_indexes", []),
        "plan_trace": state["plan_steps"],
    }
