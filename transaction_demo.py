from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from threading import Event, Thread

import neo4j
from dotenv import load_dotenv


load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DB_NAME = os.getenv("NEO4J_DATABASE", "neo4j")

DEMO_RUN_ID = f"demo-{uuid.uuid4().hex[:8]}"
DEMO_PATIENT_ID_BASE = 9_000_000

META_RUN_ID = "demo_run_id"
META_TX_LABEL = "tx_label"


def tx_meta(label: str, **extra: str) -> dict[str, str]:
    """Standard metadata for any demo transaction."""
    return {META_RUN_ID: DEMO_RUN_ID, META_TX_LABEL: label, **extra}


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def step(msg: str) -> None:
    print(f"  -> {msg}")


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    print(f"  [{label}] took {time.perf_counter() - t0:.3f}s")


def demo_autocommit(driver: neo4j.Driver) -> None:
    banner("1. Auto-commit transaction (session.run)")
    pid = DEMO_PATIENT_ID_BASE + 1
    note = "Auto-commit demo: low back pain, no red flags."

    with driver.session(database=DB_NAME) as session:
        session.run(
            """
            MERGE (p:PATIENT:DemoNode {patient_id: $pid})
            SET p.clinician_note = $note,
                p.demo_run_id    = $run_id
            """,
            pid=pid, note=note, run_id=DEMO_RUN_ID,
        )
        step(f"Inserted demo PATIENT {pid} via auto-commit.")

        result = session.run(
            "MATCH (p:PATIENT {patient_id: $pid}) RETURN p.clinician_note AS note",
            pid=pid,
        )
        step(f"Readback: {result.single()['note']!r}")


def demo_explicit_commit(driver: neo4j.Driver) -> None:
    banner("2. Explicit transaction with commit")
    pid = DEMO_PATIENT_ID_BASE + 2
    note = "Explicit-commit demo: clinician requests T2 SAG follow-up."

    with driver.session(database=DB_NAME) as session:
        tx = session.begin_transaction()
        try:
            tx.run(
                """
                MERGE (p:PATIENT:DemoNode {patient_id: $pid})
                SET p.clinician_note = $note,
                    p.demo_run_id    = $run_id
                """,
                pid=pid, note=note, run_id=DEMO_RUN_ID,
            )
            step("PATIENT staged in tx (not yet visible to other sessions).")

            tx.run(
                """
                MATCH (p:PATIENT {patient_id: $pid})
                MERGE (i:IMAGE:DemoNode {
                    patient_id:   $pid,
                    instance_uid: $iuid
                })
                SET i.series_description = 't2_sag_demo',
                    i.image_link        = $link,
                    i.demo_run_id       = $run_id
                MERGE (p)-[:HAS_IMAGE]->(i)
                """,
                pid=pid,
                iuid=f"{DEMO_RUN_ID}-iuid-A",
                link=f"http://localhost:9000/mri-ima/{pid}/demo.ima",
                run_id=DEMO_RUN_ID,
            )
            step("IMAGE + HAS_IMAGE relationship staged in same tx.")

            tx.commit()
            step("tx.commit() -> both writes now durable in one atomic step.")
        finally:
            tx.close()  # no-op if already committed; safety net otherwise


def demo_explicit_rollback(driver: neo4j.Driver) -> None:
    banner("3. Explicit transaction with rollback (atomicity)")
    pid = DEMO_PATIENT_ID_BASE + 3

    with driver.session(database=DB_NAME) as session:
        tx = session.begin_transaction()
        try:
            tx.run(
                """
                MERGE (p:PATIENT:DemoNode {patient_id: $pid})
                SET p.clinician_note = 'Should not survive rollback.',
                    p.demo_run_id    = $run_id
                """,
                pid=pid, run_id=DEMO_RUN_ID,
            )
            tx.run(
                """
                MERGE (i:IMAGE:DemoNode {
                    patient_id:   $pid,
                    instance_uid: $iuid
                })
                SET i.demo_run_id = $run_id
                """,
                pid=pid,
                iuid=f"{DEMO_RUN_ID}-rollback-iuid",
                run_id=DEMO_RUN_ID,
            )
            step("Two writes staged. Now simulating an application error...")
            raise RuntimeError("Simulated failure -- triggering rollback")
        except RuntimeError as e:
            tx.rollback()
            step(f"Caught: {e}. tx.rollback() issued.")
        finally:
            tx.close()

        # Verify nothing was persisted.
        result = session.run(
            """
            MATCH (n:DemoNode)
            WHERE (n:PATIENT AND n.patient_id = $pid)
               OR (n:IMAGE   AND n.patient_id = $pid)
            RETURN count(n) AS leftover
            """,
            pid=pid,
        )
        leftover = result.single()["leftover"]
        step(f"Leftover demo nodes for patient {pid} after rollback: {leftover}")
        assert leftover == 0, "Rollback failed: data persisted unexpectedly!"

def demo_managed_write(driver: neo4j.Driver) -> None:
    banner("4. Managed write transaction (execute_write)")

    def upsert_patient_with_image(tx, pid: int, note: str, iuid: str) -> int:
        tx.run(
            """
            MERGE (p:PATIENT:DemoNode {patient_id: $pid})
            SET p.clinician_note = $note,
                p.demo_run_id    = $run_id
            """,
            pid=pid, note=note, run_id=DEMO_RUN_ID,
        )
        tx.run(
            """
            MATCH (p:PATIENT {patient_id: $pid})
            MERGE (i:IMAGE:DemoNode {
                patient_id:   $pid,
                instance_uid: $iuid
            })
            SET i.series_description = 't1_tra_demo',
                i.image_link        = $link,
                i.demo_run_id       = $run_id
            MERGE (p)-[:HAS_IMAGE]->(i)
            """,
            pid=pid, iuid=iuid,
            link=f"http://localhost:9000/mri-ima/{pid}/{iuid}.ima",
            run_id=DEMO_RUN_ID,
        )
        result = tx.run(
            """
            MATCH (p:PATIENT {patient_id: $pid})-[:HAS_IMAGE]->(i:IMAGE)
            RETURN count(i) AS n
            """,
            pid=pid,
        )
        return result.single()["n"]

    pid = DEMO_PATIENT_ID_BASE + 4
    with driver.session(database=DB_NAME) as session:
        count = session.execute_write(
            upsert_patient_with_image,
            pid=pid,
            note="Managed-write demo: idempotent upsert.",
            iuid=f"{DEMO_RUN_ID}-managed-iuid",
        )
        step(f"Managed tx returned image count for patient {pid}: {count}")
        step("Re-running same callback to show idempotency...")
        count2 = session.execute_write(
            upsert_patient_with_image,
            pid=pid,
            note="Managed-write demo: idempotent upsert.",
            iuid=f"{DEMO_RUN_ID}-managed-iuid",
        )
        step(f"Image count after second run (should still be 1): {count2}")

def demo_managed_read(driver: neo4j.Driver) -> None:
    banner("5. Managed read transaction (execute_read)")

    def fetch_demo_patients(tx):
        result = tx.run(
            """
            MATCH (p:PATIENT:DemoNode {demo_run_id: $run_id})
            OPTIONAL MATCH (p)-[:HAS_IMAGE]->(i:IMAGE)
            RETURN p.patient_id      AS pid,
                   p.clinician_note  AS note,
                   collect(i.image_link) AS images
            ORDER BY pid
            """,
            run_id=DEMO_RUN_ID,
        )
        return [dict(record) for record in result]

    with driver.session(database=DB_NAME) as session:
        rows = session.execute_read(fetch_demo_patients)
        step(f"Found {len(rows)} demo patients created so far.")
        for row in rows:
            preview = (row["note"] or "")[:60]
            step(f"  pid={row['pid']} images={len(row['images'])} note={preview!r}")


def demo_batched_transactions(driver: neo4j.Driver) -> None:
    banner("6. Batched transactions (CALL { ... } IN TRANSACTIONS)")
    base = DEMO_PATIENT_ID_BASE + 100
    n_rows = 50
    batch_size = 10

    rows = [
        {
            "pid": base + i,
            "note": f"Batched demo patient {i}",
            "iuid": f"{DEMO_RUN_ID}-bulk-{i}",
            "link": f"http://localhost:9000/mri-ima/{base + i}/bulk.ima",
            "run_id": DEMO_RUN_ID,
        }
        for i in range(n_rows)
    ]

    query = """
    UNWIND $rows AS row
    CALL (row) {
        MERGE (p:PATIENT:DemoNode {patient_id: row.pid})
        SET p.clinician_note = row.note,
            p.demo_run_id    = row.run_id
        MERGE (i:IMAGE:DemoNode {
            patient_id:   row.pid,
            instance_uid: row.iuid
        })
        SET i.image_link  = row.link,
            i.demo_run_id = row.run_id
        MERGE (p)-[:HAS_IMAGE]->(i)
    } IN TRANSACTIONS OF $batch_size ROWS
    """

    with driver.session(database=DB_NAME) as session, timed("batched insert"):
        session.run(query, rows=rows, batch_size=batch_size).consume()
        step(f"Inserted {n_rows} patient+image pairs in batches of {batch_size}.")

        count = session.run(
            """
            MATCH (p:PATIENT:DemoNode {demo_run_id: $run_id})
            WHERE p.patient_id >= $lo AND p.patient_id < $hi
            RETURN count(p) AS n
            """,
            run_id=DEMO_RUN_ID, lo=base, hi=base + n_rows,
        ).single()["n"]
        step(f"Verified count of bulk patients in DB: {count}")


def demo_timeout_and_metadata(driver: neo4j.Driver) -> None:
    banner("7. Transaction timeout & metadata")

    with driver.session(database=DB_NAME) as session:
        tx = session.begin_transaction(
            timeout=5.0,  # seconds; server aborts if exceeded
            metadata=tx_meta(
                "timeout-demo",
                app="transaction_demo",
                purpose="patient/image read",
            ),
        )
        try:
            result = tx.run(
                """
                MATCH (p:PATIENT:DemoNode {demo_run_id: $run_id})
                RETURN count(p) AS n
                """,
                run_id=DEMO_RUN_ID,
            )
            step(f"Demo patients visible from tagged tx: {result.single()['n']}")
            tx.commit()
            step("Tx committed. Metadata was visible to ops while it ran.")
        except Exception:
            tx.rollback()
            raise


def show_active_demo_txs(driver: neo4j.Driver, label: str) -> None:
    cypher = f"""
    SHOW TRANSACTIONS
    YIELD transactionId, currentQuery, status, elapsedTime,
          metaData, activeLockCount
    WHERE metaData['{META_RUN_ID}'] = $rid
    RETURN transactionId               AS tid,
           metaData['{META_TX_LABEL}'] AS tx_label,
           status                      AS status,
           activeLockCount             AS locks,
           toString(elapsedTime)       AS elapsed,
           substring(coalesce(currentQuery, ''), 0, 70) AS query
    ORDER BY tx_label
    """
    try:
        with driver.session(database=DB_NAME) as session:
            rows = session.execute_read(
                lambda tx: [dict(r) for r in tx.run(cypher, rid=DEMO_RUN_ID)]
            )
    except neo4j.exceptions.ClientError as e:
        # SHOW TRANSACTIONS requires admin privileges on some setups.
        step(f"[live tx panel] SHOW TRANSACTIONS unavailable: {e.message}")
        return

    print(f"  --- live SHOW TRANSACTIONS @ {label} ---")
    if not rows:
        print("      (no demo-tagged transactions currently active)")
    else:
        header = f"      {'tx_label':<10} {'status':<10} {'locks':>5} {'elapsed':<14} query"
        print(header)
        print("      " + "-" * (len(header) - 6))
        for r in rows:
            print(
                f"      {str(r['tx_label'] or '-'):<10} "
                f"{r['status']:<10} "
                f"{r['locks']:>5} "
                f"{r['elapsed']:<14} "
                f"{r['query']!r}"
            )
    print()


def _read_note_fresh_session(driver: neo4j.Driver, pid: int) -> str | None:
    with driver.session(database=DB_NAME) as session:
        rec = session.execute_read(
            lambda tx: tx.run(
                "MATCH (p:PATIENT {patient_id: $pid}) RETURN p.clinician_note AS n",
                pid=pid,
            ).single()
        )
        return rec["n"] if rec else None


def _seed_patient(driver: neo4j.Driver, pid: int, note: str) -> None:
    with driver.session(database=DB_NAME) as session:
        session.execute_write(
            lambda tx: tx.run(
                """
                MERGE (p:PATIENT:DemoNode {patient_id: $pid})
                SET p.clinician_note = $note,
                    p.demo_run_id    = $run_id
                """,
                pid=pid, note=note, run_id=DEMO_RUN_ID,
            )
        )


def demo_inflight_visibility(driver: neo4j.Driver) -> None:
    banner("8. In-flight visibility: writer's view vs outsider's view")
    pid = DEMO_PATIENT_ID_BASE + 5
    _seed_patient(driver, pid, "initial")
    step(f"Seeded patient {pid} with note='initial' (committed).")

    # Coordinate writer + outsider via an Event so we don't race on a
    # fixed sleep on slow machines.
    staged = Event()

    def writer():
        with driver.session(database=DB_NAME) as session:
            tx = session.begin_transaction(
                metadata=tx_meta("writer"),
            )
            try:
                tx.run(
                    "MATCH (p:PATIENT {patient_id: $pid}) "
                    "SET p.clinician_note = 'updated-by-writer'",
                    pid=pid,
                )
                staged.set()
                step("[writer] SET applied INSIDE tx; not committed yet.")

                # The writer can already see its own change because it
                # queries through the same open transaction.
                own = tx.run(
                    "MATCH (p:PATIENT {patient_id: $pid}) RETURN p.clinician_note AS n",
                    pid=pid,
                ).single()["n"]
                step(f"[writer] self-read inside same tx  -> {own!r}")

                # Hold the tx open long enough for the outsider to observe.
                time.sleep(1.2)

                tx.commit()
                step("[writer] COMMIT issued.")
            except Exception:
                tx.rollback()
                raise

    t = Thread(target=writer, daemon=True)
    t.start()

    # Wait until the writer has staged its SET (but not committed).
    if not staged.wait(timeout=5.0):
        raise RuntimeError("writer thread didn't stage within 5s")
    show_active_demo_txs(driver, label="writer staged, not committed")
    outside_before = _read_note_fresh_session(driver, pid)
    step(f"[outsider] read while writer's tx still open -> {outside_before!r}")
    step("            ^ read-committed: outsider does NOT see uncommitted data.")

    t.join()  # wait for the commit to land
    show_active_demo_txs(driver, label="after writer commit")

    outside_after = _read_note_fresh_session(driver, pid)
    step(f"[outsider] read AFTER writer commit          -> {outside_after!r}")
    assert outside_before == "initial"
    assert outside_after == "updated-by-writer"


def demo_concurrent_same_node(driver: neo4j.Driver) -> None:
    banner("9. Two concurrent transactions on the SAME patient (lock contention)")
    pid = DEMO_PATIENT_ID_BASE + 6
    _seed_patient(driver, pid, "initial")
    step(f"Seeded patient {pid} with note='initial'.")

    a_holds_lock = Event()
    timings: dict[str, float] = {}

    def tx_a():
        with driver.session(database=DB_NAME) as session:
            tx = session.begin_transaction(metadata=tx_meta("TX-A"))
            try:
                tx.run(
                    "MATCH (p:PATIENT {patient_id: $pid}) "
                    "SET p.clinician_note = 'from-tx-A'",
                    pid=pid,
                )
                a_holds_lock.set()
                step("[TX-A] SET applied; holding write-lock for 1.0s.")
                time.sleep(1.0)
                tx.commit()
                timings["a_commit"] = time.perf_counter()
                step("[TX-A] COMMIT.")
            except Exception:
                tx.rollback()
                raise

    def tx_b():
        # Wait until TX-A has the lock so we're guaranteed to contend.
        a_holds_lock.wait()
        with driver.session(database=DB_NAME) as session:
            tx = session.begin_transaction(metadata=tx_meta("TX-B"))
            try:
                step("[TX-B] about to SET on the same patient (will block)...")
                t_call = time.perf_counter()
                tx.run(
                    "MATCH (p:PATIENT {patient_id: $pid}) "
                    "SET p.clinician_note = 'from-tx-B'",
                    pid=pid,
                )
                timings["b_unblocked"] = time.perf_counter()
                step(
                    f"[TX-B] SET returned after waiting "
                    f"{timings['b_unblocked'] - t_call:.2f}s (TX-A had the lock)."
                )
                tx.commit()
                step("[TX-B] COMMIT.")
            except Exception:
                tx.rollback()
                raise

    ta = Thread(target=tx_a, daemon=True)
    tb = Thread(target=tx_b, daemon=True)
    ta.start()
    tb.start()
    # Snapshot the server's view while TX-A holds the lock and TX-B is
    # blocked waiting. TX-B should appear with status=Blocked.
    a_holds_lock.wait(timeout=5.0)
    time.sleep(0.2)  # let TX-B reach its SET and start blocking
    show_active_demo_txs(driver, label="TX-A holds lock, TX-B blocked")
    ta.join()
    tb.join()
    show_active_demo_txs(driver, label="after both committed")

    final = _read_note_fresh_session(driver, pid)
    step(f"Final committed note for patient {pid}: {final!r}")
    if "a_commit" in timings and "b_unblocked" in timings:
        gap = timings["b_unblocked"] - timings["a_commit"]
        step(f"TX-B was unblocked ~{gap*1000:.0f}ms after TX-A's commit.")
    step("Takeaway: writers on the same node are SERIALIZED, not lost.")


#
def demo_concurrent_different_nodes(driver: neo4j.Driver) -> None:
    banner("10. Two concurrent transactions on DIFFERENT patients (true parallelism)")
    pid_x = DEMO_PATIENT_ID_BASE + 7
    pid_y = DEMO_PATIENT_ID_BASE + 8
    _seed_patient(driver, pid_x, "initial-X")
    _seed_patient(driver, pid_y, "initial-Y")
    step(f"Seeded patients {pid_x} ('initial-X') and {pid_y} ('initial-Y').")

    def worker(label: str, pid: int, new_note: str, iuid: str):
        with driver.session(database=DB_NAME) as session:
            tx = session.begin_transaction(
                metadata=tx_meta(label),
            )
            try:
                tx.run(
                    "MATCH (p:PATIENT {patient_id: $pid}) "
                    "SET p.clinician_note = $note",
                    pid=pid, note=new_note,
                )
                tx.run(
                    """
                    MATCH (p:PATIENT {patient_id: $pid})
                    MERGE (i:IMAGE:DemoNode {
                        patient_id:   $pid,
                        instance_uid: $iuid
                    })
                    SET i.series_description = 'parallel_demo',
                        i.image_link        = $link,
                        i.demo_run_id       = $run_id
                    MERGE (p)-[:HAS_IMAGE]->(i)
                    """,
                    pid=pid, iuid=iuid,
                    link=f"http://localhost:9000/mri-ima/{pid}/{iuid}.ima",
                    run_id=DEMO_RUN_ID,
                )
                step(f"[{label}] writes staged on patient {pid}; holding 0.8s.")
                time.sleep(0.8)
                tx.commit()
                step(f"[{label}] COMMIT.")
            except Exception:
                tx.rollback()
                raise

    t0 = time.perf_counter()
    tx = Thread(target=worker, args=("TX-X", pid_x, "from-tx-X",
                                     f"{DEMO_RUN_ID}-parallel-X"), daemon=True)
    ty = Thread(target=worker, args=("TX-Y", pid_y, "from-tx-Y",
                                     f"{DEMO_RUN_ID}-parallel-Y"), daemon=True)
    tx.start(); ty.start()
    # Snapshot while both txs are mid-flight. Both should show status=Running
    # with non-zero activeLockCount on different nodes, proving they overlap.
    time.sleep(0.3)
    show_active_demo_txs(driver, label="both txs running in parallel")
    tx.join(); ty.join()
    wall = time.perf_counter() - t0

    final_x = _read_note_fresh_session(driver, pid_x)
    final_y = _read_note_fresh_session(driver, pid_y)
    step(f"Final note for patient {pid_x}: {final_x!r}")
    step(f"Final note for patient {pid_y}: {final_y!r}")
    step(f"Wall-clock for both txs: {wall:.2f}s "
         f"(serialized would be ~1.6s, parallel ~0.8s).")
    step("Takeaway: disjoint write-sets -> no lock contention -> real parallelism.")


def _count_real_patients(driver: neo4j.Driver) -> int:
    cypher = """
    MATCH (p:PATIENT)
    WHERE NOT p:DemoNode
    RETURN count(p) AS n
    """
    with driver.session(database=DB_NAME) as session:
        return session.execute_read(
            lambda tx: tx.run(cypher).single()["n"]
        )


def demo_real_data(driver: neo4j.Driver) -> None:
    banner("11. Read-only managed transactions against REAL ingested data")

    n_real = _count_real_patients(driver)
    if n_real == 0:
        step("No real PATIENT nodes found (only :DemoNode patients exist).")
        step("Skipping. To populate real data, run:")
        step("  python push_data_into_minio.py && python import_data.py")
        return
    step(f"Found {n_real} real :PATIENT nodes (not :DemoNode). Querying...")

  
    def pick_patient_with_image(tx):
        result = tx.run(
            """
            MATCH (p:PATIENT)-[:HAS_IMAGE]->(i:IMAGE)
            WHERE NOT p:DemoNode AND NOT i:DemoNode
            RETURN p.patient_id AS pid, count(i) AS n_images
            ORDER BY n_images DESC, pid ASC
            LIMIT 1
            """
        )
        return result.single()


    def fetch_patient_bundle(tx, pid: int):
        result = tx.run(
            """
            MATCH (p:PATIENT {patient_id: $pid})
            OPTIONAL MATCH (p)-[:HAS_IMAGE]->(i:IMAGE)
            WITH p, i
            ORDER BY i.series_description
            RETURN p.clinician_note                     AS note,
                   collect({
                       link:   i.image_link,
                       series: i.series_description,
                       labels: [l IN labels(i) WHERE l <> 'IMAGE']
                   })                                   AS images
            """,
            pid=pid,
        )
        return result.single()

    def vector_search_neighbours(tx, pid: int, k: int = 5):
        result = tx.run(
            """
            MATCH (p:PATIENT {patient_id: $pid})-[:HAS_IMAGE]->(seed:IMAGE)
            WHERE seed.image_embedding IS NOT NULL
            WITH seed LIMIT 1
            CALL db.index.vector.queryNodes(
                'image_embedding_index', $k, seed.image_embedding
            ) YIELD node AS img, score
            RETURN seed.instance_uid AS seed_uid,
                   img.patient_id    AS pid,
                   img.instance_uid  AS uid,
                   img.image_link    AS link,
                   score
            ORDER BY score DESC
            """,
            pid=pid, k=k,
        )
        return [dict(r) for r in result]

    with driver.session(database=DB_NAME) as session:
        rec = session.execute_read(pick_patient_with_image)
        if rec is None:
            step("No real patient has linked images yet; only notes are loaded.")
            step("Run `python import_data.py` to ingest images.")
            return
        pid = rec["pid"]
        step(f"Selected real patient_id={pid} ({rec['n_images']} linked images).")

        bundle = session.execute_read(fetch_patient_bundle, pid=pid)
        note_preview = (bundle["note"] or "")[:200].replace("\n", " ")
        step(f"clinician_note (truncated): {note_preview!r}")
        step(f"Image count for patient {pid}: {len(bundle['images'])}")
        for img in bundle["images"][:3]:
            step(f"  - labels={img['labels']}  series={img['series']!r}")
            step(f"    link={img['link']}")
        if len(bundle["images"]) > 3:
            step(f"  ... +{len(bundle['images']) - 3} more.")

        try:
            neighbours = session.execute_read(
                vector_search_neighbours, pid=pid, k=5
            )
        except neo4j.exceptions.ClientError as e:
            step(f"Vector search skipped (no image_embedding_index?): {e.message}")
            return

    if not neighbours:
        step("No embeddings present on this patient's images; skipping vector search.")
        return
    step(f"Top-{len(neighbours)} similar images to seed {neighbours[0]['seed_uid']}:")
    for row in neighbours:
        marker = " (seed)" if row["uid"] == row["seed_uid"] else ""
        step(f"  score={row['score']:.4f}  pid={row['pid']}  uid={row['uid']}{marker}")


def cleanup(driver: neo4j.Driver) -> None:
    banner("Cleanup: removing all nodes tagged with this run id")
    with driver.session(database=DB_NAME) as session:
        session.run(
            """
            MATCH (n:DemoNode {demo_run_id: $run_id})
            CALL (n) {
                DETACH DELETE n
            } IN TRANSACTIONS OF 500 ROWS
            """,
            run_id=DEMO_RUN_ID,
        ).consume()
        leftover = session.run(
            "MATCH (n:DemoNode {demo_run_id: $run_id}) RETURN count(n) AS n",
            run_id=DEMO_RUN_ID,
        ).single()["n"]
        step(f"Demo nodes remaining for run {DEMO_RUN_ID}: {leftover}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
        raise SystemExit(
            "Missing Neo4j credentials. Set NEO4J_URI / NEO4J_USERNAME / "
            "NEO4J_PASSWORD in your .env file."
        )

    driver = neo4j.GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    print(f"Connected to Neo4j ({NEO4J_URI}). Demo run id: {DEMO_RUN_ID}")

    try:
        demo_autocommit(driver)
        demo_explicit_commit(driver)
        demo_explicit_rollback(driver)
        demo_managed_write(driver)
        demo_managed_read(driver)
        demo_batched_transactions(driver)
        demo_timeout_and_metadata(driver)
        demo_inflight_visibility(driver)
        demo_concurrent_same_node(driver)
        demo_concurrent_different_nodes(driver)
        demo_real_data(driver)
    finally:
        cleanup(driver)
        driver.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
