from datasets import load_dataset
import pymongo


db = pymongo.MongoClient()['climate']
ds = load_dataset("tdiggelm/climate_fever")

evidences = []
for record in ds['test']:
    for evidence in record['evidences']:
        evidence['claim_id'] = record['claim_id']
        evidence['claim'] = record['claim']
        evidence['claim_label'] = record['claim_id']
        evidences.append(evidence)

db['evidences'].insert_many(evidences)        