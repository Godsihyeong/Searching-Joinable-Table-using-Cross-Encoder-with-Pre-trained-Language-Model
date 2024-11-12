import os
import ast
import umap
import numpy as np
import pandas as pd

from tqdm import tqdm

path = './embed_table/'
output_path = './reduced_table/'

os.makedirs(output_path, exist_ok=True)

dataset = os.listdir(path)

umap_model = umap.UMAP(n_components=30, n_neighbors=10, random_state=35)

for data in tqdm(dataset, desc='Processing'):
    table = pd.read_csv(path + data)
    
    if table.shape[0] < 30:
        print(f"Skipping {data} as it has less than 30 rows.")
        continue
    
    if 'Unnamed: 0' in table.columns:
        table = table.drop('Unnamed: 0', axis = 1)
    
    if 'embedded_Unnamed: 0' in table.columns:
        table = table.drop('embedded_Unnamed: 0', axis = 1)
    
    all_embeddings = []
    
    for embedding in table.values.reshape(-1):
        if pd.notna(embedding):
            all_embeddings.append(ast.literal_eval(embedding))
        else:
            all_embeddings.append([0] * 256)
    
    table_reduced = umap_model.fit_transform(np.array(all_embeddings))
    
    np.save(f'{output_path}reduced_{data[4:-4]}.npy', table_reduced)
