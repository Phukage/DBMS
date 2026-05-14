# Neo4j Graph Database Projects

This repository is a graph database project built with Neo4j:

 **Medical Imaging RAG System** - A multimodal retrieval system for lumbar spine MRI data with vector similarity search


## Table of Contents

- [Overview](#overview)
- [Medical Imaging RAG System](#medical-imaging-rag-system)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

---

## Overview

This repository demonstrates graph database applications using Neo4j, including:
- Vector embeddings for multimodal search (images + text)
- Storing and querying text and image data
- MinIO object storage integration for medical images

---

## Medical Imaging RAG System

### Description

A multimodal Retrieval-Augmented Generation (RAG) system for lumbar spine MRI data. The system enables semantic search across both radiologist notes (text) and medical images using BiomedCLIP embeddings.

### Features

- **Multimodal Embeddings**: Uses BiomedCLIP (Microsoft) for both image and text embeddings
- **Vector Search**: Semantic similarity search on both clinician notes and MRI images
- **DICOM Processing**: Automatic extraction of metadata from medical images
- **Object Storage**: MinIO integration for image storage and retrieval
- **Graph Relationships**: Links patients to their images with rich metadata

### Data Schema

**Nodes:**
- `PATIENT`: Patient records with clinician notes and text embeddings
- `IMAGE`: MRI images with DICOM metadata and image embeddings

**Relationships:**
- `HAS_IMAGE`: Connects patients to their medical images

**Labels on Images:**
- `T1`, `T2`: MRI sequence types
- `SAG`, `TRA`: Image orientations (sagittal, transverse)

### Key Files

- `import_data.py`: Main data ingestion pipeline for patients and images
- `push_data_into_minio.py`: Uploads DICOM images to MinIO object storage
- `query.py`: Sample Cypher queries for retrieval
- `transaction_demo.py`: Walks through Neo4j transaction-control patterns (auto-commit, explicit commit/rollback, managed read/write, batched `CALL ... IN TRANSACTIONS`, timeouts/metadata, in-flight visibility, concurrent same-node lock contention, concurrent disjoint-node parallelism). Prints a live `SHOW TRANSACTIONS` panel at each interesting checkpoint, and runs an extra read-only block against the real ingested data when present.
- `loader/`: Dockerfile + entrypoint for the opt-in compose `loader` service that ingests the real dataset inside the compose network.
- `image_dataset_explore.ipynb`: Exploratory analysis of image dataset
- `text_dataset_explore.ipynb`: Exploratory analysis of radiologist notes

### Running the transaction demo against a local Neo4j

The repo ships a `docker-compose.yml` that brings up **Neo4j** (query log
turned all the way up) and **MinIO** (with the `mri-ima` bucket pre-created
and set public-read) together — so the transaction demo *and* the full
data ingestion (`push_data_into_minio.py` + `import_data.py`) work against
the same local stack with no extra setup.

```bash
docker compose up -d
```

Point your `.env` at the compose stack:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
NEO4J_DATABASE=neo4j
```

Run the demo and, in other terminals, tail the logs. Each concurrency-demo
tx is tagged with `tx_label` + `demo_run_id` in its `metaData`, which
appears verbatim in `query.log` so you can correlate log lines to demo
steps:

```bash
# terminal A
python transaction_demo.py

# terminal B -- every Cypher statement with parameters + tx metadata
docker compose exec neo4j tail -F /logs/query.log

# terminal C -- lock waits / deadlocks / lifecycle
docker compose exec neo4j tail -F /logs/debug.log
```

The demo also prints a live `SHOW TRANSACTIONS` panel at each interesting
checkpoint (writer staged but not committed, lock contention, parallel
execution), so you can see in-flight transactions directly without
leaving the terminal.

#### Loading the real dataset

The repo also ships an opt-in `loader` service that runs both ingestion
scripts inside the compose network (so it talks to `neo4j:7687` and
`minio:9000` directly, while still storing host-reachable image URLs).
It is gated behind the `load` profile so it never runs as part of a plain
`docker compose up`.

Place your dataset at:

```
./dataset/01_MRI_Data/...
./dataset/Radiologists Notes for Lumbar Spine MRI Dataset/Radiologists Report.xlsx
```

Then run the loader once:

```bash
docker compose --profile load up loader
```

First run builds the loader image (pulls the CPU build of PyTorch + 
`open_clip_torch`) and downloads the BiomedCLIP weights into a named
`hf-cache` volume; subsequent runs reuse both. Containerised PyTorch on
Apple Silicon is CPU-only, so for fastest ingestion you can still run
the scripts directly on the host instead:

```bash
python push_data_into_minio.py   # uploads DICOM files to MinIO
python import_data.py            # ingests into Neo4j with embeddings
```

Both paths produce the same graph — pick whichever you prefer.

#### Demo 11 against the real data

Once the dataset is loaded (by either path), re-running
`python transaction_demo.py` automatically picks up the real
`:PATIENT` / `:IMAGE` nodes for a final read-only block: it joins
`clinician_note` with `image_link`s for a real patient and runs a
vector-similarity query against the `image_embedding_index`, all inside
managed `execute_read` transactions. If the database has no real data,
the block prints a hint and skips itself.

Tear down:

```bash
docker compose down        # stop, keep volumes (data preserved)
docker compose down -v     # stop + wipe all volumes (clean slate)
```

### Sample Queries

**Query 1: Get patient notes and all images**
```cypher
MATCH (p:PATIENT {patient_id: $patient_id})
OPTIONAL MATCH (p)-[:HAS_IMAGE]->(i:IMAGE)
RETURN p.clinician_note AS clinician_note,
       collect(i.image_link) AS images;
```

**Query 2: Vector similarity search on images**
```cypher
MATCH (p:PATIENT {patient_id: $patient_id})-[:HAS_IMAGE]->(i:IMAGE:T2:SAG {instance_uid: $instance_uid})
CALL db.index.vector.queryNodes("image_embedding_index", $limit, i.image_embedding)
YIELD node AS img, score
RETURN img.image_link AS images, score;
```

### Embedding Models

Uses a single multimodal model for both text and images:

- Model: BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- Text Encoder: PubMedBERT (biomedical language model)
- Image Encoder: ViT-B/16 (Vision Transformer)
- Embedding Dimensions: 512 (shared vector space)
- Similarity Function: Cosine similarity

---

## Prerequisites

- Python 3.8+
- Neo4j Database (Aura or local instance)
- MinIO Server (for medical imaging project)
- CUDA-compatible GPU (optional, for faster embedding generation)

### Python Dependencies

```
neo4j
pandas
python-dotenv
tqdm
pydicom
torch
numpy
open_clip_torch
pillow
minio
openpyxl
```

---

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd <repository-name>
```

2. **Install Python dependencies**
```bash
pip install neo4j pandas python-dotenv tqdm pydicom torch numpy open_clip_torch pillow sentence-transformers minio openpyxl
```

3. **Set up Neo4j + MinIO with the bundled compose file**

The repo ships a `docker-compose.yml` that brings up both Neo4j (with the
query log fully enabled, used by `transaction_demo.py`) and MinIO (with
the `mri-ima` bucket pre-created and set public-read) in one command:

```bash
docker compose up -d
```

Alternative: use a managed Neo4j Aura instance and a standalone MinIO of
your own — in that case just configure the `.env` (next section) to point
at them.

---

## Configuration

1. **Create a `.env` file** in the project root:

```env
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
AURA_INSTANCEID=your-instance-id
AURA_INSTANCENAME=Instance01
```

2. **Configure MinIO** (medical imaging project):
- Default credentials: `minioadmin` / `minioadmin`
- Accessible at: `http://localhost:9000`

---

## Usage

### Medical Imaging RAG System

1. **Prepare your dataset**:
   - Place radiologist notes in: `./dataset/Radiologists Notes for Lumbar Spine MRI Dataset/Radiologists Report.xlsx`
   - Place DICOM images in: `./dataset/01_MRI_Data/`

2. **Upload images to MinIO**:
```bash
python push_data_into_minio.py
```

3. **Ingest data into Neo4j**:
```bash
python import_data.py
```

This will:
- Process DICOM images and extract metadata
- Generate embeddings for images using BiomedCLIP
- Generate embeddings for radiologist notes
- Create PATIENT and IMAGE nodes
- Establish relationships
- Create vector indexes

4. **Explore the data**:
```bash
jupyter notebook image_dataset_explore.ipynb
jupyter notebook text_dataset_explore.ipynb
```

---

## Project Structure

```
.
├── import_data.py                  # Medical data ingestion
├── push_data_into_minio.py        # MinIO upload script
├── query.py                        # Sample Cypher queries
├── explore/                       
    ├── image_dataset_explore.ipynb    # Image data exploration
    └── text_dataset_explore.ipynb     # Text data exploration  
├── _env                            # Environment configuration template
└── dataset/                        # Data directory (not in repo)
    ├── 01_MRI_Data/               # DICOM images
    ├── Radiologists Notes.../     # Excel file with notes
    └── processed_data/            # Cached embeddings
```

---

## Technologies Used

### Core Technologies
- **Neo4j**: Graph database with vector search capabilities
- **Python**: Primary programming language
- **MinIO**: S3-compatible object storage for medical images

### Machine Learning
- **BiomedCLIP**: Multimodal medical imaging model (Microsoft)
- **OpenCLIP**: Open-source CLIP implementation
- **PyTorch**: Deep learning framework

### Data Processing
- **Pandas**: Data manipulation and analysis
- **PyDICOM**: DICOM medical image processing
- **tqdm**: Progress bars for batch operations

### Database
- **Neo4j Python Driver**: Official Neo4j client
- **Vector Indexes**: For semantic similarity search
- **Cypher**: Graph query language

---


### Medical Imaging System
- Batch size: 1000 records per transaction
- Vector dimensions: 512
- Processed data is cached as Parquet files for faster re-runs

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in `import_data.py`
   - Use CPU instead: Remove GPU device selection

2. **MinIO Connection Failed**:
   - Ensure MinIO is running on port 9000
   - Check credentials in `push_data_into_minio.py`

3. **Neo4j Connection Timeout**:
   - Verify `.env` credentials
   - Check firewall settings for ports 7687 (Bolt) and 7474 (HTTP)

4. **Missing DICOM Files**:
   - Ensure dataset directory structure matches expected paths
   - Check file extensions (should be `.ima`)

---

## License

<!-- This project is provided as-is for educational and research purposes.

**Note**: Medical data handling should comply with HIPAA, GDPR, and other relevant regulations. The football data is sourced from public datasets. -->

---

## Acknowledgments

- BiomedCLIP model by Microsoft Research
- Neo4j for graph database technology