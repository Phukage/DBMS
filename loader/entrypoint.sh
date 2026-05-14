#!/bin/sh
# Loader entrypoint: wait for services, then run the two ingestion scripts.
# Service URLs are taken from the compose-provided env:
#   NEO4J_URI            = bolt://neo4j:7687
#   MINIO_ENDPOINT       = minio:9000           (where the loader UPLOADS to)
#   MINIO_PUBLIC_BASE    = http://localhost:9000 (what gets stored in DB)
set -eu

echo "[loader] waiting for neo4j to accept Bolt..."
# python -c is enough; we already have the neo4j driver installed.
python - <<'PY'
import os, time, neo4j
uri = os.environ["NEO4J_URI"]
user = os.environ["NEO4J_USERNAME"]
pw   = os.environ["NEO4J_PASSWORD"]
for i in range(60):
    try:
        d = neo4j.GraphDatabase.driver(uri, auth=(user, pw))
        d.verify_connectivity()
        d.close()
        print(f"[loader] neo4j ready after {i*2}s")
        break
    except Exception as e:
        print(f"[loader] neo4j not ready yet ({e.__class__.__name__}); retry in 2s")
        time.sleep(2)
else:
    raise SystemExit("neo4j did not become ready in time")
PY

echo "[loader] waiting for minio to be live..."
i=0
until curl -fsS "http://${MINIO_ENDPOINT}/minio/health/live" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -ge 60 ]; then
        echo "[loader] minio did not become ready in time" >&2
        exit 1
    fi
    sleep 2
done
echo "[loader] minio ready"

if [ ! -d /app/dataset/01_MRI_Data ] \
   || [ ! -f "/app/dataset/Radiologists Notes for Lumbar Spine MRI Dataset/Radiologists Report.xlsx" ]; then
    echo "[loader] ./dataset is missing expected files. Mount your dataset at"
    echo "         /app/dataset/01_MRI_Data and"
    echo "         /app/dataset/Radiologists Notes for Lumbar Spine MRI Dataset/Radiologists Report.xlsx"
    exit 2
fi

echo "[loader] pushing DICOM images to MinIO -> ${MINIO_ENDPOINT}"
python /app/push_data_into_minio.py

echo "[loader] ingesting patients + images into Neo4j -> ${NEO4J_URI}"
python /app/import_data.py

echo "[loader] done."
