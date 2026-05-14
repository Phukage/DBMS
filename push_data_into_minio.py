from minio import Minio
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

file_path = Path("./dataset/01_MRI_Data")
img_files = []
for root, dirs, files in os.walk(file_path):
    # print(root, dirs, files)
    if files != []:
        img_files = img_files + [Path(os.path.join(root, file)) for file in files]

print(len(img_files), "files found.")

# Endpoint can be overridden via env so this script works both from the
# host (defaults to localhost:9000) and from inside the docker-compose
# `loader` container (minio:9000).
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)

bucket = "mri-ima"

# Create bucket if it doesn't exist
if not client.bucket_exists(bucket):
    client.make_bucket(bucket)
    
# set the bucket to be public by using the following command in terminal:
# mc alias set local http://127.0.0.1:9000 minioadmin minioadmin
# mc anonymous set download local/mri-images

for img in tqdm(img_files, desc="Pushing images to MinIO", total=len(img_files)):
    client.fput_object(
        bucket,
        f"{img.parts[2]}/{img.parts[-1]}",     # object name inside bucket
        img # local file
)

print("Uploaded successfully!")
