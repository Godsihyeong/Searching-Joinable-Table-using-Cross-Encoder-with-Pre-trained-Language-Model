import os
import numpy as np
import pandas as pd
import hdbscan

from diversity_score import clustering, diversity_score
from tqdm import tqdm

path = './reduced_table/'

table = pd.read_csv('./search_results.csv', header=None, names=['Query', 'idx', 'Similarity', 'Target'])

query_tables = table['Query']
target_tables = table['Target']

scores = []

i = 0

for query in tqdm(query_tables, desc = 'Processing...'):
    
    try:
        qry = np.load(path + 'reduced_' + query[:-3] + 'npy')
    except:
        scores.append(None)
        continue
    
    for target in target_tables:
        
        print(f'Processing Table = {target}')
        
        if query == target:
            scores.append(None)
            continue
        
        try:
            trg = np.load(path + 'reduced_' + target[:-3] + 'npy')
            
            cluster_labels = clustering(qry, trg)
            
            div_score = diversity_score(cluster_labels)
            
            scores.append(div_score)
            
            i += 1
            print(f'Succed in {round(((i / table.shape[0]) * 100), 3)}')
            
            print('===== <Seperate Line> =====')
            
        except:
            scores.append(None)
            
table['diversity'] = scores

table.to_csv('final_results.csv', index=False)

print('All Done')