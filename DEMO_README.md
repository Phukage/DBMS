# Concurrency & Recovery Demos — Branch README

This branch adds [`concurrency_recovery_demo.ipynb`](./concurrency_recovery_demo.ipynb): a Jupyter notebook that maps Neo4j's transaction-control and concurrency-control behavior to Elmasri & Navathe ch. 21–22, run against the lumbar-spine MRI subset loaded by `load_subset.py`.

Every demo prints the relevant slice of `query.log` or `debug.log` directly inside its cell — no terminal-tailing required.

## What's in the notebook

| # | Theme       | Demo                                  | Maps to (Elmasri & Navathe) |
|---|-------------|---------------------------------------|-----------------------------|
| 1 | Transaction | Atomic multi-entity rollback          | ch. 21 — atomicity          |
| 2 | Transaction | Checkpoint + WAL truncation           | ch. 22 — checkpointing      |
| 3 | Transaction | Crash recovery via WAL replay         | ch. 22 — log-based recovery |
| 4 | Concurrency | Non-repeatable read (READ COMMITTED)  | ch. 21 — isolation levels   |
| 5 | Concurrency | Lost update (naive RMW vs atomic SET) | ch. 21 — lost-update anomaly|


A `.env` file (in `code/`) pointing at the local stack:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
NEO4J_DATABASE=neo4j
```

## One-time setup

```bash
docker compose up -d
python load_subset.py # load preproces data of 10 patients
```

After step 2 the database holds 10 real `:PATIENT` nodes (`[27, 35, 99, 117, 127, 158, 251, 275, 308, 475]`) and ~1107 `:IMAGE` nodes wired up with `:HAS_IMAGE`, plus the two uniqueness constraints (`PATIENT.patient_id`, `IMAGE.instance_uid`) and the two vector indexes.

If you don't yet have the subset parquet files, regenerate them from the full dataset:

```bash
python make_subset.py
```
