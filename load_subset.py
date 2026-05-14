"""
Load the precomputed subset (produced by `make_subset.py`) into Neo4j.

Reads two parquet files and:
  1. creates uniqueness constraints on PATIENT.patient_id and IMAGE.instance_uid
  2. UNWIND-MERGE upserts the patients + their image nodes
  3. creates HAS_IMAGE relationships
  4. tags images with T1/T2/SAG/TRA labels from series_description
  5. creates 512-dim cosine vector indexes for both embeddings

Idempotent: rerunning leaves the same graph.

Usage:
  python load_subset.py                          # reads subset/ at repo root
  python load_subset.py --in-dir other/dir
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import neo4j
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
DB_NAME = os.getenv("NEO4J_DATABASE", "neo4j")
BATCH_SIZE = 500


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def load(driver, patients_df: pd.DataFrame, images_df: pd.DataFrame) -> None:
    with driver.session(database=DB_NAME) as session:
        # --- 1. constraints (make MERGE O(1) instead of label scan) ---------
        print("Creating constraints...")
        session.run(
            "CREATE CONSTRAINT patient_id IF NOT EXISTS "
            "FOR (p:PATIENT) REQUIRE p.patient_id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT image_uid IF NOT EXISTS "
            "FOR (i:IMAGE) REQUIRE i.instance_uid IS UNIQUE"
        )

        # --- 2. patients ----------------------------------------------------
        patient_rows = patients_df.to_dict("records")
        print(f"Upserting {len(patient_rows)} PATIENT nodes...")
        session.run(
            """
            UNWIND $rows AS r
            MERGE (p:PATIENT {patient_id: r.patient_id})
            SET p.clinician_note = r.clinician_note,
                p.note_embedding = r.note_embedding
            """,
            rows=patient_rows,
        )

        # --- 3. images + HAS_IMAGE relationships ----------------------------
        image_rows = images_df.to_dict("records")
        # `SET i += r` would also overwrite instance_uid with itself (fine)
        # and copy patient_id onto the IMAGE node, which the existing schema
        # uses for the labelling/relationship step. Keep all DICOM metadata.
        print(f"Upserting {len(image_rows)} IMAGE nodes in batches of {BATCH_SIZE}...")
        for batch in chunks(image_rows, BATCH_SIZE):
            session.run(
                """
                UNWIND $rows AS r
                MERGE (i:IMAGE {instance_uid: r.instance_uid})
                SET i += r
                WITH i, r
                MATCH (p:PATIENT {patient_id: r.patient_id})
                MERGE (p)-[:HAS_IMAGE]->(i)
                """,
                rows=batch,
            )

        # --- 4. T1/T2/SAG/TRA labels from series_description ---------------
        print("Tagging images with T1 / T2 / SAG / TRA labels...")
        session.run(
            """
            MATCH (img:IMAGE)
            WHERE img.series_description IS NOT NULL
            WITH img, toLower(img.series_description) AS desc
            FOREACH (_ IN CASE WHEN desc CONTAINS 't1'  THEN [1] ELSE [] END | SET img:T1)
            FOREACH (_ IN CASE WHEN desc CONTAINS 't2'  THEN [1] ELSE [] END | SET img:T2)
            FOREACH (_ IN CASE WHEN desc CONTAINS 'sag' THEN [1] ELSE [] END | SET img:SAG)
            FOREACH (_ IN CASE WHEN desc CONTAINS 'tra' THEN [1] ELSE [] END | SET img:TRA)
            """
        )

        # --- 5. vector indexes ---------------------------------------------
        print("Creating 512-d cosine vector indexes...")
        session.run(
            """
            CREATE VECTOR INDEX image_embedding_index IF NOT EXISTS
            FOR (i:IMAGE) ON i.image_embedding
            OPTIONS { indexConfig: {
                `vector.dimensions`: 512,
                `vector.similarity_function`: 'cosine'
            }}
            """
        )
        session.run(
            """
            CREATE VECTOR INDEX note_embedding_index IF NOT EXISTS
            FOR (p:PATIENT) ON p.note_embedding
            OPTIONS { indexConfig: {
                `vector.dimensions`: 512,
                `vector.similarity_function`: 'cosine'
            }}
            """
        )

        # --- summary -------------------------------------------------------
        counts = session.run(
            """
            MATCH (p:PATIENT) WITH count(p) AS patients
            MATCH (i:IMAGE)   WITH patients, count(i) AS images
            MATCH ()-[r:HAS_IMAGE]->() RETURN patients, images, count(r) AS rels
            """
        ).single()
        print(
            f"Done. Graph now has {counts['patients']} PATIENT, "
            f"{counts['images']} IMAGE, {counts['rels']} HAS_IMAGE."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="subset")
    args = parser.parse_args()
    in_dir = Path(args.in_dir)

    patients_path = in_dir / "patients.parquet"
    images_path = in_dir / "images.parquet"
    if not patients_path.exists() or not images_path.exists():
        raise SystemExit(
            f"Missing parquet files in {in_dir}. Run `python make_subset.py` first."
        )

    print(f"Reading {patients_path} and {images_path}...")
    patients_df = pd.read_parquet(patients_path)
    images_df = pd.read_parquet(images_path)
    print(f"  patients={len(patients_df)} rows, images={len(images_df)} rows")

    driver = neo4j.GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    print(f"Connected to Neo4j ({NEO4J_URI}).")
    try:
        load(driver, patients_df, images_df)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
