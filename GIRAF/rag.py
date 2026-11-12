import os
import numpy as np
import config


_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(getattr(config, "RAG_EMBED_MODEL", "BAAI/bge-m3"))
        _MODEL.max_seq_length = 2048
    return _MODEL


def _split_md(text, source):
    chunks = []
    cur_heading = None
    cur_lines = []
    for line in text.splitlines():
        if line.startswith("## "):
            if cur_heading is not None:
                chunks.append({
                    "source": source,
                    "heading": cur_heading,
                    "text": "\n".join(cur_lines).strip(),
                })
            cur_heading = line[3:].strip()
            cur_lines = [line]
        elif line.startswith("# "):
            continue
        else:
            if cur_heading is not None:
                cur_lines.append(line)
    if cur_heading is not None:
        chunks.append({
            "source": source,
            "heading": cur_heading,
            "text": "\n".join(cur_lines).strip(),
        })
    return chunks


def _experience_chunk(fn, text):
    heading = "General experience"
    body = []
    for line in text.splitlines():
        if line.startswith("# ") and heading == "General experience":
            heading = line[2:].strip()
            continue
        body.append(line)
    return {
        "source": fn[:-3],
        "heading": heading,
        "text": "\n".join(body).strip(),
        "kind": "experience",
    }


def load_chunks(kb_dir):
    kb_dir = kb_dir or getattr(config, "RAG_KB_DIR", None)
    chunks = []
    for fn in sorted(os.listdir(kb_dir)):
        if not fn.endswith(".md"):
            continue
        with open(os.path.join(kb_dir, fn), encoding="utf-8") as f:
            text = f.read()
        if fn.startswith("workload"):
            for ch in _split_md(text, fn[:-3]):
                ch["kind"] = "heuristic"
                chunks.append(ch)
        else:
            chunks.append(_experience_chunk(fn, text))
    return chunks


def embed_texts(texts):
    model = _get_model()
    try:
        vecs = model.encode(list(texts), convert_to_numpy=True, batch_size=16)
    except Exception:
        vecs = model.to("cpu").encode(list(texts), convert_to_numpy=True, batch_size=16)
    vecs = vecs.astype("float32")
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


_CHUNK_EMB_CACHE = {}


def embed_chunks(chunks):
    if not chunks:
        return np.zeros((0, 0), dtype="float32")
    key = tuple((c["source"], c["heading"], c["text"]) for c in chunks)
    if key not in _CHUNK_EMB_CACHE:
        _CHUNK_EMB_CACHE[key] = embed_texts([c["text"] for c in chunks])
    return _CHUNK_EMB_CACHE[key]


_PLAN = "Execution plan tree"
_HEURISTIC = "heuristic optimal solution"


def strip_plan(text):
    ip = text.find(_PLAN)
    if ip == -1:
        return text
    head = text[:ip].rstrip()
    ib = text.find(_HEURISTIC, ip)
    if ib == -1:
        return head
    return head + "\n\n" + text[ib:].strip()


def format_references(refs, include_plan=False):
    if not refs:
        return ""
    blocks = []
    for r in refs:
        text = r["text"] if include_plan else strip_plan(r["text"])
        blocks.append(f"Source: {r['source']} | {r['heading']}\n{text}")
    return "\n\n".join(blocks)


def retrieve_references2(workload_text, k=None):
    k = int(k or getattr(config, "RAG_TOP_K", 4))
    chunks = load_chunks(getattr(config, "RAG_KB_DIR", None))
    query = (workload_text or "")
    exp = sorted((c for c in chunks if c["kind"] == "experience"), key=lambda c: c["source"])
    heuristic = [c for c in chunks if c["kind"] == "heuristic"]

    q = embed_texts([query])[0]
    c = embed_chunks(heuristic)
    sims = c @ q
    k = min(int(k), len(heuristic))
    idx = np.argsort(-sims)[:k]
    retrieved = [{**heuristic[i], "score": float(sims[i])} for i in idx]
    refs = list(exp) + retrieved
    for i, r in enumerate(refs):
        r["ref_id"] = f"R{i + 1}"
    return refs
