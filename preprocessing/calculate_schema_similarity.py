import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from schema_similarity import compatibility, cosine_similarity, schema_similarity

compatibility_dict = {'object':{'object':1.0, 'float32':0.3, 'float64':0.3, 'int32':0.3, 'int64':0.3, 'bool':0.5},
                 'float32':{'object':0.3, 'float32':1.0, 'float64':1.0, 'int32':0.9, 'int64':0.9, 'bool':0.1},
                 'float64':{'object':0.3, 'float32':1.0, 'float64':1.0, 'int32':0.9, 'int64':0.9, 'bool':0.1},
                 'int32':{'object':0.3, 'float32':0.9, 'float64':0.9, 'int32':1.0, 'int64':1.0, 'bool':0.1},
                 'int64':{'object':0.3, 'float32':0.9, 'float64':0.9, 'int32':1.0, 'int64':1.0, 'bool':0.1},
                 'bool':{'object':0.5, 'float32':0.1, 'float64':0.1, 'int32':0.1, 'int64':0.1, 'bool':1.0}}

table_path = os.path.join(os.getcwd(), 'dataset')
col_path = os.path.join(os.getcwd(), 'columns_vector')

query = []
target = []
schema_sim = []

dataset = os.listdir(col_path)

for i in tqdm(range(len(dataset)), desc = 'Processing'):

    df1 = pd.read_csv(os.path.join(table_path, dataset[i][:-3] + 'csv'))   
    col_vector1 = np.load(os.path.join(col_path, dataset[i]))

    for j in range(i+1, len(dataset)):
        
        df2 = pd.read_csv(os.path.join(table_path, dataset[j][:-3] + 'csv'))
        col_vector2 = np.load(os.path.join(col_path, dataset[j]))
        # 1
        compatibility_matrix = compatibility(df1.dtypes.tolist(), df2.dtypes.tolist(), compatibility_dict)
        # 2     
        col_similarity_matrix = cosine_similarity(col_vector1, col_vector2)
        # 3
        result = schema_similarity(0.7, compatibility_matrix, col_similarity_matrix)

        query.append(dataset[i][:-4])
        target.append(dataset[j][:-4])
        schema_sim.append(result)
        
schema_sim_df = pd.DataFrame({'Query':query, 'Target':target, 'Schema_sim':schema_sim})
schema_sim_df.to_csv('schema_sim.csv', index=False)