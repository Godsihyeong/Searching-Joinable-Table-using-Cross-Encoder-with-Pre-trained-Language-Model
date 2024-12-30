# import os
# import pandas as pd
# import numpy as np
# from tqdm import tqdm

# from schema_similarity import compatibility, cosine_similarity, schema_similarity

# compatibility_dict = {'object':{'object':1.0, 'float32':0.3, 'float64':0.3, 'int32':0.3, 'int64':0.3, 'bool':0.5},
#                  'float32':{'object':0.3, 'float32':1.0, 'float64':1.0, 'int32':0.9, 'int64':0.9, 'bool':0.1},
#                  'float64':{'object':0.3, 'float32':1.0, 'float64':1.0, 'int32':0.9, 'int64':0.9, 'bool':0.1},
#                  'int32':{'object':0.3, 'float32':0.9, 'float64':0.9, 'int32':1.0, 'int64':1.0, 'bool':0.1},
#                  'int64':{'object':0.3, 'float32':0.9, 'float64':0.9, 'int32':1.0, 'int64':1.0, 'bool':0.1},
#                  'bool':{'object':0.5, 'float32':0.1, 'float64':0.1, 'int32':0.1, 'int64':0.1, 'bool':1.0}}

# table_path = os.path.join(os.getcwd(), 'dataset')

# table_embed_path = os.path.join(os.getcwd(), 'input_table')

# query = []
# target = []
# schema_sim = []

# dataset = os.listdir(table_embed_path)

# for i in tqdm(range(len(dataset)), desc = 'Processing'):

#     df1 = pd.read_csv(os.path.join(table_path, dataset[i][:-3] + 'csv'))   
#     table_vector1 = np.load(os.path.join(table_embed_path, dataset[i]))
#     column_mean1 = np.mean(table_vector1, axis = 1)
    
#     for j in range(i+1, len(dataset)):
        
#         df2 = pd.read_csv(os.path.join(table_path, dataset[j][:-3] + 'csv'))
#         table_vector2 = np.load(os.path.join(table_embed_path, dataset[j]))
#         column_mean2 = np.mean(table_vector2, axis = 1)
        
#         # 1     
#         indices, col_similarity_matrix = cosine_similarity(column_mean1, column_mean2, threshold=0.7)
#         # 2
#         compatibility_matrix = compatibility(df1.dtypes.tolist(), df2.dtypes.tolist(), compatibility_dict, indices)
#         # 3
#         result = schema_similarity(compatibility_matrix, col_similarity_matrix, alpha=0.7)

#         query.append(dataset[i][:-4])
#         target.append(dataset[j][:-4])
#         schema_sim.append(result)
        
# schema_sim_df = pd.DataFrame({'Query':query, 'Target':target, 'Schema_sim_with_col_datas':schema_sim})
# schema_sim_df.to_csv('schema_sim_with_col_datas.csv', index=False)

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