# Concurrency & Recovery Demos

Notebook [`concurrency_recovery_demo.ipynb`](./concurrency_recovery_demo.ipynb) — Neo4j behavior mapped to Elmasri & Navathe ch. 21–22, run against the 10-patient subset.

| # | Theme       | Demo                              |
|---|-------------|-----------------------------------|
| 1 | Transaction | Atomic multi-entity rollback      |
| 2 | Transaction | Checkpoint + WAL truncation       |
| 3 | Transaction | Crash recovery via WAL replay     |
| 4 | Concurrency | Non-repeatable read               |
| 5 | Concurrency | Lost update (naive vs atomic SET) |

Every demo prints the relevant slice of `query.log` or `debug.log` in-cell — no terminal-tailing needed. Demos 4 and 5 use `Event`/`Barrier` synchronization so outcomes are deterministic.

## Setup

```bash
cd code
docker compose up -d           # start neo4j-demo
python load_subset.py          # load 10-patient subset
jupyter notebook concurrency_recovery_demo.ipynb
```

`.env` (in `code/`):

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
NEO4J_DATABASE=neo4j
```

Run cells top-to-bottom. Setup is idempotent — re-running it wipes prior demo state.

## What each cell verifies (deterministic outputs)

| Demo | Assertion |
|------|-----------|
| 1    | `before == after` after a forced `ConstraintError` rollback |
| 2    | `Checkpoint started … completed` lines in `debug.log` |
| 3    | `:CrashMarker` is queryable after `SIGKILL` + restart; `Recovery completed` in `debug.log` |
| 4    | `r1 != r2` from two reads in the same READER transaction |
| 5    | naive final == 100 (lost 100); atomic final == 200 |

## Cleanup

```bash
docker compose down            # stop, keep volumes
docker compose down -v         # stop + wipe everything
```

## Troubleshooting

| Symptom                                                  | Fix                                                            |
|----------------------------------------------------------|----------------------------------------------------------------|
| Setup raises `No :PATIENT nodes`                         | Run `python load_subset.py`.                                   |
| Demo 2 says `db.checkpoint() unavailable`                | Expected on Community edition; the 8 s wait still picks up the scheduled checkpoint. |
| Demo 3 hangs                                             | `docker logs neo4j-demo`.                                      |
| Log slice prints `(no matches)`                          | Re-run the cell; pattern didn't hit any fresh lines.           |

## Files

```
code/
├── concurrency_recovery_demo.ipynb   # the notebook
├── DEMO_README.md                    # this file
├── load_subset.py                    # upserts the subset into Neo4j
├── subset/{patients,images}.parquet  # 10 patients, ~1107 images
└── docker-compose.yml                # neo4j-demo + minio-demo
```
