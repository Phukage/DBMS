import neo4j
from dotenv import load_dotenv
import os

load_dotenv()
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_URI = os.getenv("NEO4J_URI")
DB_NAME = "neo4j"
BATCH_SIZE = 1000

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
driver.verify_connectivity()
print("Connected to Neo4j instance successfully!")

# Retrieve patient.note and image of that patient when given a patient id
query_1 = """
MATCH (p:PATIENT {patient_id: $patient_id})
OPTIONAL MATCH (p)-[:HAS_IMAGE]->(i:IMAGE)
RETURN p.clinician_note AS clinician_note,
       collect(i.image_link) AS images;
"""


query_2 = """
MATCH (p:PATIENT {patient_id: $patient_id})
OPTIONAL MATCH (p)-[:HAS_IMAGE]->(i:IMAGE)
WHERE $condition
RETURN p.clinician_note AS clinician_note,
       collect(i.image_link) AS images;
"""

query_3 = """
MATCH (p:PATIENT {patient_id: $patient_id}) -[:HAS_IMAGE]->(i:IMAGE:T2:SAG {instance_uid: $instance_uid})
CALL db.index.vector.queryNodes("image_embedding_index", $limit, i.image_embedding)
YIELD node AS img, score
RETURN img.image_link AS image_link, score;
"""

def execute_query_1(id):
    # id is the patient_id which is an int 
    with driver.session(database = DB_NAME) as session:
        result = session.run(query_1, id=id)
        for record in result:
            print(f"Patient ID: {record['patient_id']}")
            print(f"Note: {record['clinician_note']}")
            print(f"Images: {record['images']}")

def execute_query_2(id, condition):
    # id is the patient_id which is an int
    # condition is a string that represents the condition to filter images, e.g. "T2" in labels(i) AND "SAG" in labels(i)
    with driver.session(database = DB_NAME) as session:
        result = session.run(query_2, id=id, condition=condition)
        for record in result:
            print(f"Patient ID: {record['patient_id']}")
            print(f"Note: {record['clinician_note']}")
            print(f"Images: {record['images']}")
            
def execute_query_3(id, instance_uid, limit = 5):
    # id is the patient_id which is an int
    # instance_uid is a string that represents the instance_uid of the image to query against
    # limit is an int that represents the number of similar images to return
    with driver.session(database = DB_NAME) as session:
        result = session.run(query_3, id=id, instance_uid=instance_uid, limit=limit)
        for record in result:
            print(f"Image: {record['image_link']}")
            print(f"Similarity Score: {record['score']}")