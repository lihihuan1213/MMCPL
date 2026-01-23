import pandas as pd
import torch

def preprocess_skill_to_resource(csv_path):
    df = pd.read_csv(csv_path)
    skill_to_resources = {}
    for skill_id, group in df.groupby('knowledge_concept_id'):
        resource_ids = list(set(group['resource_id'].tolist()))
        skill_to_resources[int(skill_id)] = resource_ids
    return skill_to_resources

