import neo4j
from dotenv import load_dotenv
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pydicom
import torch
import numpy as np


load_dotenv()
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_URI = os.getenv("NEO4J_URI")
DB_NAME = "neo4j"
BATCH_SIZE = 1000

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
driver.verify_connectivity()
print("Connected to Neo4j instance successfully!")


text_data_path = Path("./dataset/Radiologists Notes for Lumbar Spine MRI Dataset/Radiologists Report.xlsx")
image_data_path = Path("./dataset/01_MRI_Data")
processed_data_dir = Path("./dataset/processed_data")



import open_clip
from PIL import Image

model, preprocess = open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

model.eval()

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

def compute_image_embedding(image_array, model, preprocess):
    image = image_array.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)

    img = Image.fromarray(image).convert("RGB")
    image_input = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    vector = image_features.squeeze().cpu().numpy().tolist()
    return vector

def get_img_data(processed_data_dir, image_data_path, model=None, preprocess=None):
    processed_data_dir.mkdir(parents = True, exist_ok=True)
    processed_image_data_path = (processed_data_dir / image_data_path.name).with_suffix(".parquet" )
    if processed_image_data_path.exists():
        image_df = pd.read_parquet(processed_image_data_path)
        return image_df.to_dict(orient="records")
    img_files = []
    for root, dirs, files in os.walk(image_data_path):
        # print(root, dirs, files)
        if files != []:
            img_files = img_files + [Path(os.path.join(root, file)) for file in files]
            
    
    img_data = []
    length = len(img_files)
    for img in tqdm(img_files, desc="Processing images", total=length):
        if img.suffix.lower() != ".ima":
            continue
        ds = pydicom.dcmread(img)
        info_dict = dicom_to_dict(ds)        
        info_dict["patient_id"] = int(img.parts[2])
        info_dict["image_link"] = f"http://localhost:9000/mri-ima/{img.parts[2]}/{img.parts[-1]}"
        info_dict["image_embedding"] = compute_image_embedding(ds.pixel_array, model, preprocess)
        img_data.append(info_dict)
    
    img_df = pd.DataFrame(img_data)
    img_df.to_parquet(processed_image_data_path, index=False)
    return img_df.to_dict(orient="records")

def dicom_to_dict(ds):

    def safe_get(tag):
        """Return None if tag missing or empty."""
        value = getattr(ds, tag, None)

        if value == "" or value == []:
            return None

        # Convert special pydicom types to python native
        if hasattr(value, "tolist"):  
            value = value.tolist()

        return value

    metadata = {
        # KEEP IDENTIFIERS
        "study_uid": safe_get("StudyInstanceUID"),
        "series_uid": safe_get("SeriesInstanceUID"),
        "instance_uid": safe_get("SOPInstanceUID"),

        # Patient (if available)
        "patient_sex": safe_get("PatientSex"),
        "patient_age": safe_get("PatientAge"),
        "patient_weight": safe_get("PatientWeight"),
        "patient_size": safe_get("PatientSize"),

        # Scanner info
        "modality": safe_get("Modality"),
        "manufacturer": safe_get("Manufacturer"),
        "model": safe_get("ManufacturerModelName"),
        "magnetic_field_strength": safe_get("MagneticFieldStrength"),

        # Study / series
        "study_description": safe_get("StudyDescription"),
        "series_description": safe_get("SeriesDescription"),
        "body_part": safe_get("BodyPartExamined"),

        # Geometry (VERY valuable for ML later)
        "slice_thickness": safe_get("SliceThickness"),
        "rows": safe_get("Rows"),
        "columns": safe_get("Columns"),

        # Acquisition parameters
        "repetition_time": safe_get("RepetitionTime"),
        "echo_time": safe_get("EchoTime"),
        "flip_angle": safe_get("FlipAngle"),

        # Position
        # "image_orientation": safe_get("ImageOrientationPatient"),

        # File reference (VERY important)
        # "file_path": str(Path(dicom_path).resolve())
    }

    return metadata

def insert_patient_img_nodes(patient_img_dict, driver, batch_size = BATCH_SIZE):
    query = """
            UNWIND $patient_img_dict AS img
            MERGE (i:IMAGE { patient_id: img.patient_id, study_uid: img.study_uid, series_uid: img.series_uid, instance_uid: img.instance_uid })
            SET
                i.patient_sex = img.patient_sex,               
                i.patient_age = img.patient_age,
                i.patient_weight = img.patient_weight,
                i.patient_size = img.patient_size,
                i.modality = img.modality,
                i.manufacturer = img.manufacturer,
                i.model = img.model,
                i.magnetic_field_strength = img.magnetic_field_strength,
                i.study_description = img.study_description,
                i.series_description = img.series_description,
                i.body_part = img.body_part,
                i.slice_thickness = img.slice_thickness,
                i.rows = img.rows,
                i.columns = img.columns,
                i.repetition_time = img.repetition_time,
                i.echo_time = img.echo_time,
                i.flip_angle = img.flip_angle,
                i.image_embedding = img.image_embedding,
                i.image_link = img.image_link
            """
    total_patient_img = len(patient_img_dict)
    with driver.session(database = DB_NAME) as session:
        total_batches = (total_patient_img + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(patient_img_dict), batch_size), total=total_batches, desc=f"Uploading {total_patient_img} patient images to Neo4j"):
            batch = patient_img_dict[i: i + batch_size]
            session.run(query, patient_img_dict=batch)
                
def create_image_patient_relationships(driver):
    query = """
    CALL () {
    MATCH (img:IMAGE)
    MATCH (p:PATIENT {patient_id: img.patient_id})
    MERGE (p)-[:HAS_IMAGE]->(img)
    } IN TRANSACTIONS
    """
    
    with driver.session(database=DB_NAME) as session:
        session.run(query)
    

def create_img_labels(driver):
    query = """
    MATCH (img:IMAGE)
WITH img, toLower(img.series_description) AS desc

FOREACH (_ IN CASE WHEN desc CONTAINS 't1' THEN [1] ELSE [] END |
    SET img:T1
)

FOREACH (_ IN CASE WHEN desc CONTAINS 't2' THEN [1] ELSE [] END |
    SET img:T2
)

FOREACH (_ IN CASE WHEN desc CONTAINS 'tra' THEN [1] ELSE [] END |
    SET img:TRA
)

FOREACH (_ IN CASE WHEN desc CONTAINS 'sag' THEN [1] ELSE [] END |
    SET img:SAG
);

    """
    
    with driver.session(database=DB_NAME) as session:
        session.run(query)
        
def create_img_embedding_index(driver):
    query = """
    CREATE VECTOR INDEX image_embedding_index IF NOT EXISTS
    FOR (i:IMAGE)
    ON i.image_embedding
    OPTIONS { indexConfig: {
        `vector.dimensions`: 512,
        `vector.similarity_function`: 'cosine'
    }}
    """
    # query = """
    # CREATE VECTOR INDEX image_embedding_index IF NOT EXISTS
    # FOR (i:IMAGE)
    # ON i.image_embedding
    # OPTIONS { indexConfig: {
    #     `vector.dimensions`: 384,
    #     `vector.similarity_function`: 'cosine'
    # }}
    # """
    with driver.session(database = DB_NAME) as session:
        session.run(query)
    




def delete_everything(driver):
    delete_query = """
    CALL () {
    MATCH (n)
    WITH n limit 15000
    DETACH DELETE n
    } IN TRANSACTIONS

    """
    with driver.session(database = DB_NAME) as session:
        session.run(delete_query)

def encode_text(text, model, tokenizer, device):
    tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze().cpu().numpy()

def encode_text_column(text_series, model, tokenizer, device, batch_size=64):
    
    texts = text_series.tolist()
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding clinician's notes"):
        
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch).to(device)

        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu())

    return torch.cat(all_embeddings).numpy()

def get_text_data(processed_data_dir, text_data_path, model):
    processed_data_dir.mkdir(parents = True, exist_ok=True)
    processed_text_data_path = (processed_data_dir / text_data_path.name).with_suffix(".parquet" )
    
    if processed_text_data_path.exists():
        text_df = pd.read_parquet(processed_text_data_path)
        return text_df.to_dict(orient="records")
    
    text_df = pd.read_excel(text_data_path)
    patient_id, note = text_df.columns 
    text_df = text_df.rename(columns= {patient_id: "id", note: "note"})
    patient_id, note = text_df.columns 
    
    text_df = text_df[text_df[note].apply(lambda x: isinstance(x, str))]
    
    embeddings = encode_text_column(
    text_df[note],
    model,
    tokenizer,
    device,
    batch_size=64)
    
    text_df["note_embedding"] = embeddings.tolist()
    
    
    text_df.to_parquet(processed_text_data_path, index=False)
    return text_df.to_dict(orient="records")

def insert_patient_nodes(patient_dict, driver, batch_size = BATCH_SIZE):
    query = """
            UNWIND $patient_dict AS patient
            MERGE (p:PATIENT { patient_id: patient.id })
            SET
                p.clinician_note = patient.note,
                p.note_embedding = patient.note_embedding
            """
    total_patient = len(patient_dict)
    with driver.session(database = DB_NAME) as session:
        total_batches = (total_patient + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(patient_dict), batch_size), total=total_batches, desc=f"Uploading {total_patient} patients to Neo4j"):
            batch = patient_dict[i: i + batch_size]
            session.run(query, patient_dict=batch)

def create_note_vector_index(driver):
    query = """
    CREATE VECTOR INDEX note_embedding_index IF NOT EXISTS
    FOR (p:PATIENT)
    ON p.note_embedding
    OPTIONS { indexConfig: {
        `vector.dimensions`: 512,
        `vector.similarity_function`: 'cosine'
    }}
    """
    # query = """
    # CREATE VECTOR INDEX note_embedding_index IF NOT EXISTS
    # FOR (p:PATIENT)
    # ON p.note_embedding
    # OPTIONS { indexConfig: {
    #     `vector.dimensions`: 384,
    #     `vector.similarity_function`: 'cosine'
    # }}
    # """
    with driver.session(database = DB_NAME) as session:
        session.run(query)
        
def query_patient_id(driver, id):
    query = """
    MATCH (p:PATIENT {patient_id: $id})
    RETURN p.patient_id AS id, p.clinician_note AS note
    """
    with driver.session(database = DB_NAME) as session:
        result = session.run(query, id=id)
        for record in result:
            print(f"Patient ID: {record['id']}")
            print(f"Note: {record['note']}")

def query_patient_note(driver, model, note):
    query = """
    CALL db.index.vector.queryNodes('note_embedding_index', 5, $note_embedding) 
    YIELD node AS p, score
    RETURN p.patient_id AS id, p.clinician_note AS note, score
    """
    note_embedding = encode_text(note, model, tokenizer, device).tolist()
    with driver.session(database = DB_NAME) as session:
        result = session.run(query, note_embedding=note_embedding)
        for record in result:
            print(f"Patient ID: {record['id']}")
            print(f"Note: {record['note']}")
            print(f"Score: {record['score']}")
            print("-----")



    

# Patient Query Example
patient_dict = get_text_data(processed_data_dir, text_data_path, model)
insert_patient_nodes(patient_dict, driver)
create_note_vector_index(driver)

# query_patient_id(driver, 15)

# condition = """Lumbosacral MRI
# Features of muscle spasm
# Dissicating   lower disc space noted
# -Central and left paracentral disc protrusion noted at L4-L5 level compressing the thecal sac and Lt exit nerve root 
# -Diffuse Disc bulges noted at the L5-S1 level , mild compressing the thecal sac and exiting nerve root.
# """
# query_patient_note(driver, model, condition)


img_dict = get_img_data(processed_data_dir, image_data_path, model, preprocess)
insert_patient_img_nodes(img_dict, driver)
create_image_patient_relationships(driver)
create_img_labels(driver)
create_img_embedding_index(driver)



# delete_everything(driver)

driver.close()