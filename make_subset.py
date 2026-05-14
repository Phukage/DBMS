"""
Build a small, easy-to-share subset of the dataset.

Picks N patients at random (seed-controlled), embeds their clinician notes
and every DICOM in their folder using BiomedCLIP, and writes two parquet
files. The output is self-contained and can be loaded into any Neo4j with
`python load_subset.py`.

Output (default: subset/ at repo root, so the files can be committed
and pushed -- they sit OUTSIDE the gitignored dataset/ tree):
  patients.parquet  -- patient_id, clinician_note, note_embedding (512-d)
  images.parquet    -- patient_id, study/series/instance UIDs, full DICOM
                       metadata, image_link, image_embedding (512-d)

Usage:
  python make_subset.py                       # 10 patients, seed=42
  python make_subset.py --n 20 --seed 7
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import pydicom
import torch
from PIL import Image
from tqdm import tqdm

TEXT_PATH = Path("dataset/Radiologists Notes for Lumbar Spine MRI Dataset/Radiologists Report.xlsx")
IMAGE_ROOT = Path("dataset/01_MRI_Data")
MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
MINIO_BUCKET = "mri-ima"
MINIO_PUBLIC_BASE = "http://localhost:9000"

DICOM_FIELDS = [
    ("study_uid", "StudyInstanceUID"),
    ("series_uid", "SeriesInstanceUID"),
    ("instance_uid", "SOPInstanceUID"),
    ("patient_sex", "PatientSex"),
    ("patient_age", "PatientAge"),
    ("patient_weight", "PatientWeight"),
    ("patient_size", "PatientSize"),
    ("modality", "Modality"),
    ("manufacturer", "Manufacturer"),
    ("model", "ManufacturerModelName"),
    ("magnetic_field_strength", "MagneticFieldStrength"),
    ("study_description", "StudyDescription"),
    ("series_description", "SeriesDescription"),
    ("body_part", "BodyPartExamined"),
    ("slice_thickness", "SliceThickness"),
    ("rows", "Rows"),
    ("columns", "Columns"),
    ("repetition_time", "RepetitionTime"),
    ("echo_time", "EchoTime"),
    ("flip_angle", "FlipAngle"),
]


def dicom_to_dict(ds: pydicom.Dataset) -> dict:
    out = {}
    for key, tag in DICOM_FIELDS:
        v = getattr(ds, tag, None)
        if v in ("", [], None):
            out[key] = None
            continue
        if hasattr(v, "tolist"):
            v = v.tolist()
        out[key] = v
    return out


def dicom_to_pil(ds: pydicom.Dataset) -> Image.Image:
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def batch_encode_images(model, preprocess, device, imgs, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(imgs), batch_size), desc="Embedding images"):
        batch = imgs[i : i + batch_size]
        x = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            feats = model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.extend(feats.cpu().numpy().tolist())
    return embeddings


def batch_encode_text(model, tokenizer, device, texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.extend(feats.cpu().numpy().tolist())
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of patients")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--out-dir", default="subset")
    parser.add_argument("--image-batch", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- pick N patients that have BOTH a clinician note and DICOMs ----------
    text_df = pd.read_excel(TEXT_PATH)
    text_df.columns = ["patient_id", "clinician_note"]
    text_df["patient_id"] = text_df["patient_id"].astype(int)
    text_df = text_df[text_df["clinician_note"].apply(lambda x: isinstance(x, str))]

    have_notes = set(text_df["patient_id"])
    have_images = {int(p.name) for p in IMAGE_ROOT.iterdir() if p.is_dir() and p.name.isdigit()}
    candidates = sorted(have_notes & have_images)
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(candidates, args.n))
    print(f"Selected patient_ids: {selected}")

    text_subset = text_df[text_df["patient_id"].isin(selected)].copy()
    text_subset = text_subset.sort_values("patient_id").reset_index(drop=True)

    # --- load model ----------------------------------------------------------
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading BiomedCLIP on {device}...")
    model, preprocess = open_clip.create_model_from_pretrained(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    model.eval().to(device)

    # --- embed clinician notes ----------------------------------------------
    print(f"Embedding {len(text_subset)} clinician notes...")
    text_subset["note_embedding"] = batch_encode_text(
        model, tokenizer, device, text_subset["clinician_note"].tolist()
    )

    # --- walk DICOMs + embed images -----------------------------------------
    image_records: list[dict] = []
    decoded_imgs: list[Image.Image] = []
    print("Reading DICOM headers + pixel data...")
    for pid in tqdm(selected, desc="Patients"):
        patient_dir = IMAGE_ROOT / f"{pid:04d}"
        for ima_path in sorted(patient_dir.rglob("*.ima")):
            ds = pydicom.dcmread(str(ima_path))
            row = dicom_to_dict(ds)
            row["patient_id"] = pid
            row["image_link"] = (
                f"{MINIO_PUBLIC_BASE}/{MINIO_BUCKET}/{pid:04d}/{ima_path.name}"
            )
            image_records.append(row)
            decoded_imgs.append(dicom_to_pil(ds))
    print(f"Found {len(image_records)} DICOMs across {len(selected)} patients.")

    embeddings = batch_encode_images(
        model, preprocess, device, decoded_imgs, batch_size=args.image_batch
    )
    for row, emb in zip(image_records, embeddings):
        row["image_embedding"] = emb

    image_df = pd.DataFrame(image_records)

    # --- write ---------------------------------------------------------------
    patients_path = out_dir / "patients.parquet"
    images_path = out_dir / "images.parquet"
    text_subset.to_parquet(patients_path, index=False)
    image_df.to_parquet(images_path, index=False)

    print(f"Wrote {patients_path}  ({patients_path.stat().st_size / 1024:.1f} KB)")
    print(f"Wrote {images_path}    ({images_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  patients: {len(text_subset)} rows")
    print(f"  images:   {len(image_df)} rows")


if __name__ == "__main__":
    main()
