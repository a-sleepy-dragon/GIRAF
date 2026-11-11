# GIRAF
an Agentic Index Recommendation Framework for Graph Databases 

## Here is our framework overview!
<img width="1159" height="870" alt="g182" src="https://github.com/user-attachments/assets/10f38b5c-316c-450a-a770-f4e25bad63f5" />

## Environment Requirements

| Component | Version | Notes |
|---|---|---|
| Python | 3.9+ (tested on 3.10) | conda recommended |
| Neo4j | Community **5.26.0** | only one user database is online at a time |
| JDK | 17 | required by Neo4j 5.x |
| LLM endpoint | DeepSeek (OpenAI-compatible) | `https://api.deepseek.com/v1` |
| Embedding model | `BAAI/bge-m3` | downloaded once from HuggingFace (local cache afterwards) |

**Python packages** 

- `neo4j` 
- `numpy` 
- `sentence-transformers` 
---

## Installation

### Step 1 — Python environment

```bash
conda create -n GIRAF python=3.10 -y
conda activate GIRAF
pip install neo4j numpy sentence-transformers
```

### Step 2 — Neo4j + JDK

Install Neo4j Community **5.26.0** and a **JDK 17**, then point Neo4j at the JDK:

```bash
export JAVA_HOME=/path/to/jdk-17/Contents/Home
/path/to/neo4j-community-5.26.0/bin/neo4j start
```

### Step 3 — Embedding model

```bash
export HF_ENDPOINT=https://hf-mirror.com
```




