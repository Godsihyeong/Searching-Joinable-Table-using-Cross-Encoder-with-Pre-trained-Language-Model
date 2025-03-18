import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# from schema_similarity import cosine_similarity, schema_similarity

table_path = os.path.join(os.getcwd(), 'dataset')

table_embed_path = os.path.join(os.getcwd(), 'input_table')

threshold = 0.7

query = []
target = []
schema_sim = []
schema_sim_sqrt = []

dataset = os.listdir(table_embed_path)

for i in tqdm(range(len(dataset)), desc = 'Processing'):

    table_vector1 = np.load(os.path.join(table_embed_path, dataset[i]))
    column_mean1 = np.mean(table_vector1, axis = 1)
    
    for j in range(i+1, len(dataset)):
        
        table_vector2 = np.load(os.path.join(table_embed_path, dataset[j]))
        column_mean2 = np.mean(table_vector2, axis = 1)
        
        # 1     
        cos_matrix = cosine_similarity(column_mean1, column_mean2)
        indices = np.argwhere(cos_matrix >= threshold)
        results = [cos_matrix[i, j] for i, j in indices]

        query.append(dataset[i][:-4])
        target.append(dataset[j][:-4])
        schema_sim.append(np.sum(results))

        if len(results) > 0:
            schema_sim_sqrt.append(np.sum(results) / np.sqrt(len(results)))
        else:
            schema_sim_sqrt.append(np.sum(results))
        
schema_sim_df = pd.DataFrame({'Query':query, 'Target':target, 'Schema_sim_with_col_datas':schema_sim, 'Schema_sim_with_col_datas_sqrt':schema_sim_sqrt})
schema_sim_df.to_csv('schema_sim_with_col_datas.csv', index=False)
